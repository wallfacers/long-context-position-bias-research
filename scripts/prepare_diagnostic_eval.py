#!/usr/bin/env python3
"""Derive locate-only and oracle diagnostic sets from position-group evaluation data."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_bias_research.io import read_jsonl, write_jsonl_atomic
from position_bias_research.synthetic_data import render_document
from position_bias_research.tokenization import TokenCounter, load_token_counter


SUPPORTED_MODES = ("locate_only", "oracle_long", "oracle_short")
LOCATE_SYSTEM_PROMPT = (
    "Use only the context. Return valid JSON with answer \"\", supporting evidence_ids, "
    "short exact evidence_quotes, and confidence. Do not answer the question."
)
ORACLE_SYSTEM_PROMPT = (
    "Use only the supplied oracle evidence to answer the question. Return valid JSON with "
    "answer, evidence_ids, short exact evidence_quotes, and confidence."
)


def csv_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(modes) - set(SUPPORTED_MODES)
    if not modes or unknown:
        raise argparse.ArgumentTypeError(
            f"modes must be chosen from {', '.join(SUPPORTED_MODES)}; unknown={sorted(unknown)}"
        )
    return modes


def split_prompt(prompt: str) -> tuple[str, str]:
    marker = "</context>\n\nQuestion: "
    if marker not in prompt or not prompt.endswith("\nResponse:"):
        raise ValueError("Prompt does not match the expected synthetic-data template")
    context, question_and_response = prompt.split(marker, 1)
    question = question_and_response.removesuffix("\nResponse:")
    return context + "</context>", question


def oracle_block(row: dict[str, Any]) -> str:
    documents = [
        {"id": evidence_id, "text": quote}
        for evidence_id, quote in zip(
            row["target"]["evidence_ids"],
            row["target"]["evidence_quotes"],
            strict=True,
        )
    ]
    return "\n\n".join(render_document(document) for document in documents)


def context_without_oracle_documents(context: str, row: dict[str, Any]) -> str:
    """Move, rather than duplicate, gold evidence for the long oracle condition."""
    reduced = context
    for evidence_id, quote in zip(
        row["target"]["evidence_ids"],
        row["target"]["evidence_quotes"],
        strict=True,
    ):
        document = render_document({"id": evidence_id, "text": quote})
        if document not in reduced:
            raise ValueError(
                f"Cannot find exact oracle document {evidence_id} in {row['sample_id']}"
            )
        reduced = reduced.replace(document, "", 1)
    while "\n\n\n" in reduced:
        reduced = reduced.replace("\n\n\n", "\n\n")
    return reduced


def derive_row(
    row: dict[str, Any], mode: str, token_counter: TokenCounter | None = None
) -> dict[str, Any]:
    if row.get("schema_version") != "position-group-v1":
        raise ValueError(f"Unexpected source schema: {row.get('schema_version')}")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported diagnostic mode: {mode}")

    derived = copy.deepcopy(row)
    source_sample_id = row["sample_id"]
    source_group_id = row["group_id"]
    source_position_label = row["position_label"]
    context, question = split_prompt(row["prompt"])
    evidence = oracle_block(row)

    derived["sample_id"] = f"{source_sample_id}@diag-{mode}"
    derived["group_id"] = f"{source_group_id}@diag-{mode}"
    derived["evaluation_mode"] = mode
    metadata = derived.setdefault("metadata", {})
    metadata.update(
        {
            "evaluation_mode": mode,
            "source_sample_id": source_sample_id,
            "source_group_id": source_group_id,
            "source_position_label": source_position_label,
            "source_target_position": row["target_position"],
            "source_actual_position": row["actual_position"],
        }
    )

    if mode == "locate_only":
        derived["system_prompt"] = LOCATE_SYSTEM_PROMPT
        derived["target"]["answer"] = ""
    elif mode == "oracle_long":
        derived["system_prompt"] = ORACLE_SYSTEM_PROMPT
        distractor_context = context_without_oracle_documents(context, row)
        derived["prompt"] = (
            f"{distractor_context}\n\nQuestion: {question}\n\n"
            f"<oracle_evidence>\n{evidence}\n</oracle_evidence>\nResponse:"
        )
        metadata["oracle_evidence_moved_from_source_context"] = True
    else:
        derived["system_prompt"] = ORACLE_SYSTEM_PROMPT
        derived["prompt"] = (
            f"<oracle_evidence>\n{evidence}\n</oracle_evidence>\n\n"
            f"Question: {question}\nResponse:"
        )
        derived["sample_id"] = f"{source_group_id}@diag-{mode}"
        derived["position_label"] = "oracle"
        derived["target_position"] = 0.0
        derived["actual_position"] = 0.0
        metadata["deduplicated_across_source_positions"] = True
    if token_counter is not None:
        derived["actual_tokens"] = token_counter.count_chat(
            derived["system_prompt"], derived["prompt"]
        )
        metadata["diagnostic_actual_tokens"] = derived["actual_tokens"]
    return derived


def derive_rows(
    rows: Iterable[dict[str, Any]],
    modes: list[str],
    token_counter: TokenCounter | None = None,
) -> Iterable[dict[str, Any]]:
    seen_oracle_groups: set[str] = set()
    for row in rows:
        for mode in modes:
            if mode == "oracle_short":
                if row["group_id"] in seen_oracle_groups:
                    continue
                seen_oracle_groups.add(row["group_id"])
            yield derive_row(row, mode, token_counter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        type=csv_modes,
        default=list(SUPPORTED_MODES),
        help=f"Comma-separated subset of: {','.join(SUPPORTED_MODES)}",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.output}")
    rows = list(read_jsonl(args.input))
    if not rows:
        raise SystemExit(f"Empty source dataset: {args.input}")
    counter = load_token_counter(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    count = write_jsonl_atomic(
        args.output, derive_rows(rows, args.modes, token_counter=counter)
    )
    print(f"Wrote {count:,} diagnostic rows to {args.output}")
    for mode in args.modes:
        expected = len({row["group_id"] for row in rows}) if mode == "oracle_short" else len(rows)
        print(f"{mode:14} expected_rows={expected:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
