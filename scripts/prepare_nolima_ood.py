#!/usr/bin/env python3
"""Convert official NoLiMa needles/haystacks into matched position groups.

The converter keeps the character, question, book slice, and nominal input
length fixed within each group.  Only the location of the exact needle span
changes.  It intentionally does not add evidence IDs to the book text, because
an artificial marker would turn evidence localization into a lexical shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from position_bias_research.chat_protocol import apply_chat_template


ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def portable_tokenizer_name(value: str) -> str:
    path = Path(value)
    return path.name if path.exists() else value


SYSTEM_PROMPT = (
    "Use only the supplied book snippet. Return valid JSON with answer, "
    "evidence_ids, evidence_quotes, and confidence. No evidence IDs are "
    "provided, so evidence_ids must be an empty list. Evidence quotes must be "
    "short exact spans copied from the snippet."
)

USER_TEMPLATE = """You will answer a question based on the following book snippet:

<book>
{haystack}
</book>

Use the information in the book snippet to answer the question. The answer may
require a strong logical or world-knowledge inference rather than literal word
matching.

Question: {question}
Response:"""


@dataclass(frozen=True)
class NoLiMaCase:
    case_id: str
    reasoning_type: str
    question_type: str
    needle_template: str
    question: str
    character_set: tuple[str, ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def replace_arguments(template: str, values: Sequence[str]) -> str:
    rendered = template
    for index, value in enumerate(values, start=1):
        rendered = rendered.replace("{" + str(index) + "}", value)
    return rendered


def expand_cases(needle_set: list[dict[str, Any]]) -> list[NoLiMaCase]:
    cases: list[NoLiMaCase] = []
    for experiment in needle_set:
        for question_type, question_template in experiment["questions"].items():
            for test_id, test in experiment["tests"].items():
                values = tuple(str(item) for item in test["input_args"])
                question = replace_arguments(question_template, values)
                needle = replace_arguments(experiment["needle"], values)
                if any(marker in question or marker in needle for marker in ("{1}", "{2}", "{3}")):
                    raise ValueError(
                        f"Unresolved placeholder in {experiment['id']}/{test_id}/{question_type}"
                    )
                cases.append(
                    NoLiMaCase(
                        case_id=f"{experiment['id']}_{test_id}_{question_type}",
                        reasoning_type=experiment.get("reasoning_type", "unknown"),
                        question_type=question_type,
                        needle_template=needle,
                        question=question,
                        character_set=tuple(experiment.get("character_set", ())),
                    )
                )
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("NoLiMa case IDs are not unique")
    return cases


def chat_token_count(tokenizer: Any, system_prompt: str, user_prompt: str) -> int:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    encoded = apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    # transformers versions may return either a flat token-id list or a
    # BatchEncoding-like mapping.  ``len(BatchEncoding)`` counts fields, not
    # tokens, which would silently turn every prompt length into 2.
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError("Expected one chat prompt")
        encoded = encoded[0]
    return len(encoded)


def position_label(position: float) -> str:
    return f"p{round(position * 100):03d}"


def insertion_index(text: str, position: float) -> int:
    if position <= 0:
        return 0
    if position >= 1:
        return len(text)
    target = round(len(text) * position)
    previous = text.rfind("\n", 0, target + 1)
    following = text.find("\n", target)
    candidates = [value for value in (previous, following) if value >= 0]
    return min(candidates, key=lambda value: abs(value - target)) if candidates else target


def render_at_position(base_text: str, needle: str, position: float) -> tuple[str, int]:
    index = insertion_index(base_text, position)
    block = "\n" + needle + "\n"
    return base_text[:index] + block + base_text[index:], index


def fit_book_slice(
    tokenizer: Any,
    book_token_ids: list[int],
    needle: str,
    question: str,
    target_tokens: int,
    positions: Sequence[float],
) -> tuple[str, dict[float, tuple[str, int, int]]]:
    """Fit one shared prefix so every positional rendering stays under budget."""

    if target_tokens < 256:
        raise ValueError("NoLiMa target lengths below 256 tokens are unsupported")
    empty_user = USER_TEMPLATE.format(haystack="", question=question)
    overhead = chat_token_count(tokenizer, SYSTEM_PROMPT, empty_user)
    needle_tokens = len(tokenizer.encode(needle, add_special_tokens=False))
    # Leave a fixed margin for BPE boundary changes around the inserted block.
    # Re-tokenizing a 32K prompt is comparatively expensive, so fit downward in
    # at most a few passes instead of trying to fill the final handful of tokens.
    budget = min(len(book_token_ids), max(1, target_tokens - overhead - needle_tokens - 64))

    rendered: dict[float, tuple[str, int, int]] = {}
    for _ in range(3):
        base_text = tokenizer.decode(book_token_ids[:budget], skip_special_tokens=True)
        rendered = {}
        maximum = 0
        for position in positions:
            context, index = render_at_position(base_text, needle, position)
            user_prompt = USER_TEMPLATE.format(haystack=context, question=question)
            count = chat_token_count(tokenizer, SYSTEM_PROMPT, user_prompt)
            rendered[position] = (context, index, count)
            maximum = max(maximum, count)
        if maximum <= target_tokens:
            return base_text, rendered
        else:
            budget -= max(1, maximum - target_tokens + 8)
            if budget <= 0:
                raise ValueError(f"Cannot fit NoLiMa prompt into {target_tokens} tokens")
    maximum = max(item[2] for item in rendered.values())
    if maximum > target_tokens:
        raise ValueError(f"Failed to fit NoLiMa prompt: {maximum} > {target_tokens}")
    return base_text, rendered


def generate_rows(
    *,
    tokenizer: Any,
    cases: Sequence[NoLiMaCase],
    books: Sequence[Path],
    lengths: Sequence[int],
    positions: Sequence[float],
    seed: int,
    needle_set_name: str,
) -> Iterable[dict[str, Any]]:
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "unknown"))
    for book in sorted(books):
        book_text = book.read_text(encoding="utf-8")
        book_hash = sha256_bytes(book_text.encode("utf-8"))
        book_token_ids = tokenizer.encode(book_text, add_special_tokens=False)
        for case in cases:
            if "{CHAR}" in case.needle_template and not case.character_set:
                raise ValueError(f"Missing character_set for {case.case_id}")
            for target_tokens in lengths:
                group_seed = stable_seed(seed, needle_set_name, book.name, case.case_id, target_tokens)
                character = (
                    random.Random(group_seed).choice(case.character_set)
                    if "{CHAR}" in case.needle_template
                    else ""
                )
                needle = case.needle_template.replace("{CHAR}", character)
                question = case.question.replace("{CHAR}", character)
                base_text, renderings = fit_book_slice(
                    tokenizer,
                    book_token_ids,
                    needle,
                    question,
                    target_tokens,
                    positions,
                )
                base_hash = sha256_bytes(base_text.encode("utf-8"))
                group_id = (
                    f"nolima-{needle_set_name}-{book.stem}-{case.case_id}-{target_tokens}"
                )
                for position in positions:
                    context, character_index, actual_tokens = renderings[position]
                    prompt = USER_TEMPLATE.format(haystack=context, question=question)
                    prefix = context[:character_index]
                    context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
                    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
                    actual_position = prefix_tokens / max(context_tokens, 1)
                    label = position_label(position)
                    sample_id = f"{group_id}@{label}"
                    target = {
                        "answer": character,
                        "evidence_ids": [],
                        "evidence_quotes": [needle],
                        "confidence": 1.0,
                    }
                    yield {
                        "schema_version": "position-group-v1",
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "split": "test",
                        "task": f"nolima_{case.question_type}",
                        "filler_type": "nolima_book",
                        "target_tokens": target_tokens,
                        "actual_tokens": actual_tokens,
                        "position_label": label,
                        "target_position": position,
                        "actual_position": actual_position,
                        "query_position": "end",
                        "system_prompt": SYSTEM_PROMPT,
                        "prompt": prompt,
                        "target": target,
                        "seed": group_seed,
                        "tokenizer": tokenizer_name,
                        "metadata": {
                            "benchmark": "NoLiMa",
                            "benchmark_split": needle_set_name,
                            "case_id": case.case_id,
                            "question_type": case.question_type,
                            "reasoning_type": case.reasoning_type,
                            "book": book.name,
                            "book_sha256": book_hash,
                            "base_context_sha256": base_hash,
                            "character": character,
                            "evidence_id_applicable": False,
                            "official_license": "Adobe Research License; noncommercial research only",
                        },
                    }


def audit_rows(
    rows: Sequence[dict[str, Any]],
    *,
    positions: Sequence[float],
    token_slack: int = 192,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No NoLiMa rows were generated")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in NoLiMa rows")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
        if row["actual_tokens"] > row["target_tokens"]:
            raise ValueError(f"Token overflow in {row['sample_id']}")
        if row["target_tokens"] - row["actual_tokens"] > token_slack:
            raise ValueError(f"Excess token slack in {row['sample_id']}")
        quote = row["target"]["evidence_quotes"][0]
        if quote not in row["prompt"]:
            raise ValueError(f"Missing gold quote in {row['sample_id']}")
        answer = row["target"]["answer"]
        if not answer or row["prompt"].count(answer) != 1:
            raise ValueError(f"Answer is not unique in {row['sample_id']}")

    expected_labels = {position_label(position) for position in positions}
    for group_id, group in groups.items():
        labels = {row["position_label"] for row in group}
        if labels != expected_labels:
            raise ValueError(f"Position coverage mismatch in {group_id}: {labels}")
        if len({json.dumps(row["target"], sort_keys=True) for row in group}) != 1:
            raise ValueError(f"Target changed across positions in {group_id}")
        if len({row["metadata"]["base_context_sha256"] for row in group}) != 1:
            raise ValueError(f"Base context changed across positions in {group_id}")

    by_task = Counter(row["task"] for row in rows)
    by_length = Counter(str(row["target_tokens"]) for row in rows)
    by_position = Counter(row["position_label"] for row in rows)
    return {
        "schema_version": "nolima-position-audit-v1",
        "status": "ok",
        "rows": len(rows),
        "groups": len(groups),
        "positions_per_group": len(positions),
        "by_task": dict(sorted(by_task.items())),
        "by_length": dict(sorted(by_length.items())),
        "by_position": dict(sorted(by_position.items())),
        "max_actual_tokens": max(row["actual_tokens"] for row in rows),
        "max_position_error": max(
            abs(row["actual_position"] - row["target_position"]) for row in rows
        ),
    }


def csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def csv_positions(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    values = [item / 100 if item > 1 else item for item in values]
    if not values or any(item < 0 or item > 1 for item in values):
        raise argparse.ArgumentTypeError("positions must be in [0,1] or percentages")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("positions must be unique")
    return values


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--needle-set", type=Path, required=True)
    parser.add_argument("--haystack-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--lengths", type=csv_ints, default=[1024, 8192, 32000])
    parser.add_argument(
        "--positions",
        type=csv_positions,
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    destinations = (args.output, args.manifest, args.audit)
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit("Refusing to overwrite: " + ", ".join(str(path) for path in existing))
    books = sorted(args.haystack_dir.glob("*.txt"))
    if not books:
        raise SystemExit(f"No .txt haystacks under {args.haystack_dir}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    needle_payload = json.loads(args.needle_set.read_text(encoding="utf-8"))
    cases = expand_cases(needle_payload)
    rows = list(
        generate_rows(
            tokenizer=tokenizer,
            cases=cases,
            books=books,
            lengths=args.lengths,
            positions=args.positions,
            seed=args.seed,
            needle_set_name=args.needle_set.stem,
        )
    )
    audit = audit_rows(rows, positions=args.positions)
    expected = len(cases) * len(books) * len(args.lengths) * len(args.positions)
    if len(rows) != expected:
        raise RuntimeError(f"Generated {len(rows)} rows; expected {expected}")

    write_jsonl_atomic(args.output, rows)
    write_json_atomic(args.audit, audit)
    manifest = {
        "schema_version": "nolima-position-manifest-v1",
        "status": "validated",
        "source": "NoLiMa (ICML 2025), official Adobe Research repository/data",
        "source_license": "Adobe Research License; noncommercial research only",
        "needle_set": portable_path(args.needle_set),
        "needle_set_sha256": sha256_file(args.needle_set),
        "haystacks": [
            {"path": portable_path(book), "sha256": sha256_file(book)} for book in books
        ],
        "tokenizer": portable_tokenizer_name(
            str(getattr(tokenizer, "name_or_path", args.tokenizer))
        ),
        "tokenizer_revision": args.tokenizer_revision,
        "seed": args.seed,
        "lengths": args.lengths,
        "positions": args.positions,
        "cases": len(cases),
        "books": len(books),
        "rows": len(rows),
        "output": portable_path(args.output),
        "output_sha256": sha256_file(args.output),
        "audit": portable_path(args.audit),
        "audit_sha256": sha256_file(args.audit),
    }
    write_json_atomic(args.manifest, manifest)
    print(
        f"Wrote {len(rows):,} rows ({len(cases)} cases × {len(books)} books × "
        f"{len(args.lengths)} lengths × {len(args.positions)} positions) to {args.output}"
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
