#!/usr/bin/env python3
"""Derive retrieval and oracle diagnostics from a frozen NoLiMa position set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_bias_research.tokenization import load_token_counter


MODES = ("locate_only", "oracle_long", "oracle_short")
LOCATE_SYSTEM_PROMPT = (
    "Use only the supplied book snippet. Return valid JSON with answer set to an "
    "empty string, evidence_ids set to an empty list, short exact evidence_quotes "
    "copied from the snippet, and confidence. Locate the supporting evidence but "
    "do not answer the question."
)
ORACLE_SYSTEM_PROMPT = (
    "Use the supplied oracle evidence to answer the question. Return valid JSON "
    "with answer, evidence_ids set to an empty list, short exact evidence_quotes "
    "copied from the oracle block, and confidence."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_prompt(prompt: str) -> tuple[str, str]:
    book_start = "<book>\n"
    book_end = "\n</book>"
    question_marker = "\nQuestion: "
    if book_start not in prompt or book_end not in prompt or question_marker not in prompt:
        raise ValueError("Prompt does not match the frozen NoLiMa template")
    before_book, remainder = prompt.split(book_start, 1)
    book, after_book = remainder.split(book_end, 1)
    question = after_book.split(question_marker, 1)[1]
    if not question.endswith("\nResponse:"):
        raise ValueError("NoLiMa prompt is missing the response suffix")
    return book, question.removesuffix("\nResponse:")


def remove_exact_needle(book: str, needle: str) -> str:
    if book.count(needle) != 1:
        raise ValueError("Gold NoLiMa needle must occur exactly once in the book")
    surrounded = f"\n{needle}\n"
    if surrounded in book:
        # The NoLiMa constructor inserts exactly this whole block at a
        # character boundary; removing the block (including both newlines)
        # recovers the shared pre-insertion book slice.
        reduced = book.replace(surrounded, "", 1)
    else:
        reduced = book.replace(needle, "", 1)
    return reduced.strip("\n")


def render_long(book: str, question: str, needle: str) -> str:
    return (
        "You will answer a question based on the following book snippet:\n\n"
        f"<book>\n{book}\n</book>\n\n"
        "Use the information in the book snippet and the explicit oracle evidence "
        "to answer the question.\n\n"
        f"Question: {question}\n\n"
        f"<oracle_evidence>\n{needle}\n</oracle_evidence>\nResponse:"
    )


def render_short(question: str, needle: str) -> str:
    return (
        f"<oracle_evidence>\n{needle}\n</oracle_evidence>\n\n"
        f"Question: {question}\nResponse:"
    )


def derive_locate(row: dict[str, Any], counter: Any) -> dict[str, Any]:
    derived = copy.deepcopy(row)
    derived["sample_id"] = f"{row['sample_id']}@diag-locate_only"
    derived["group_id"] = f"{row['group_id']}@diag-locate_only"
    derived["evaluation_mode"] = "locate_only"
    derived["system_prompt"] = LOCATE_SYSTEM_PROMPT
    derived["target"]["answer"] = ""
    derived["actual_tokens"] = counter.count_chat(
        derived["system_prompt"], derived["prompt"]
    )
    derived.setdefault("metadata", {}).update(
        {
            "evaluation_mode": "locate_only",
            "source_sample_id": row["sample_id"],
            "source_group_id": row["group_id"],
            "source_position_label": row["position_label"],
            "diagnostic_actual_tokens": derived["actual_tokens"],
        }
    )
    return derived


def derive_oracle(row: dict[str, Any], mode: str, counter: Any) -> dict[str, Any]:
    if mode not in ("oracle_long", "oracle_short"):
        raise ValueError(mode)
    derived = copy.deepcopy(row)
    book, question = split_prompt(row["prompt"])
    needle = row["target"]["evidence_quotes"][0]
    reduced = remove_exact_needle(book, needle)
    derived["sample_id"] = f"{row['group_id']}@diag-{mode}"
    derived["group_id"] = f"{row['group_id']}@diag-{mode}"
    derived["evaluation_mode"] = mode
    derived["system_prompt"] = ORACLE_SYSTEM_PROMPT
    derived["prompt"] = (
        render_long(reduced, question, needle)
        if mode == "oracle_long"
        else render_short(question, needle)
    )
    derived["position_label"] = "oracle"
    derived["target_position"] = 1.0
    derived["actual_position"] = 1.0
    derived["actual_tokens"] = counter.count_chat(
        derived["system_prompt"], derived["prompt"]
    )
    derived.setdefault("metadata", {}).update(
        {
            "evaluation_mode": mode,
            "source_sample_id": row["sample_id"],
            "source_group_id": row["group_id"],
            "source_position_label": row["position_label"],
            "oracle_evidence_moved_to_end": True,
            "deduplicated_across_source_positions": True,
            "diagnostic_actual_tokens": derived["actual_tokens"],
        }
    )
    return derived


def derive_rows(rows: list[dict[str, Any]], counter: Any) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    output = [derive_locate(row, counter) for row in rows]
    for group_id in sorted(groups):
        group = sorted(groups[group_id], key=lambda row: row["position_label"])
        source = group[0]
        # The needle is the only positional change. Removing it must recover the
        # same distractor book for every member before oracle inputs are deduped.
        recovered = set()
        for row in group:
            book, _ = split_prompt(row["prompt"])
            recovered.add(
                remove_exact_needle(book, row["target"]["evidence_quotes"][0])
            )
        if len(recovered) != 1:
            raise ValueError(f"Recovered NoLiMa books differ across {group_id}")
        output.append(derive_oracle(source, "oracle_long", counter))
        output.append(derive_oracle(source, "oracle_short", counter))
    return output


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {args.output}")
    with args.input.open(encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    if len(source) != 1050 or len({row["group_id"] for row in source}) != 150:
        raise SystemExit("Expected the frozen 1,050-row / 150-group NoLiMa gate")
    counter = load_token_counter(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    rows = derive_rows(source, counter)
    counts = Counter(row["evaluation_mode"] for row in rows)
    expected = {"locate_only": 1050, "oracle_long": 150, "oracle_short": 150}
    if dict(counts) != expected or len({row["sample_id"] for row in rows}) != len(rows):
        raise SystemExit(f"Diagnostic completeness check failed: {dict(counts)}")
    for row in rows:
        quote = row["target"]["evidence_quotes"][0]
        if row["prompt"].count(quote) != 1:
            raise SystemExit(f"Gold quote support check failed: {row['sample_id']}")
        if row["actual_tokens"] > 32768:
            raise SystemExit(f"Diagnostic exceeds 32K: {row['sample_id']}")
    write_jsonl_atomic(args.output, rows)
    manifest = {
        "schema_version": "nolima-diagnostic-manifest-v1",
        "status": "validated",
        "source": str(args.input),
        "source_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(rows),
        "source_groups": 150,
        "mode_counts": dict(counts),
        "oracle_deduplication": "one row per position-equivalence group",
        "tokenizer": counter.name,
        "tokenizer_revision": args.tokenizer_revision,
        "license": "Adobe Research License; noncommercial research only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows):,} NoLiMa diagnostic rows: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
