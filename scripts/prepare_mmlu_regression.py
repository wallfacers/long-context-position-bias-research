#!/usr/bin/env python3
"""Prepare the pinned full MMLU test split as a short-context regression check."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ID = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
LETTERS = "ABCD"
SYSTEM_PROMPT = (
    "Return only one JSON object with keys answer, evidence_ids, evidence_quotes, and "
    "confidence. Put exactly one option letter (A, B, C, or D) in answer and empty arrays "
    "in both evidence fields. Do not add Markdown or explanation."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_for(row: dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"], strict=True)
    )
    return (
        "Choose the single best answer to the following multiple-choice question.\n\n"
        f"Question: {row['question']}\n{options}\n\n"
        "Return the option letter only in the JSON answer field."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required to prepare MMLU") from exc

    dataset = load_dataset(DATASET_ID, "all", split="test", revision=DATASET_REVISION)
    output_rows: list[dict[str, Any]] = []
    for index, source in enumerate(dataset):
        row = dict(source)
        subject = str(row["subject"])
        answer_index = int(row["answer"])
        if len(row["choices"]) != 4 or not 0 <= answer_index < 4:
            raise SystemExit(f"Invalid MMLU row at index {index}")
        sample_id = f"mmlu/{subject}/{index:05d}"
        output_rows.append(
            {
                "schema_version": "position-group-v1",
                "sample_id": sample_id,
                "group_id": sample_id,
                "split": "regression_test",
                "task": f"mmlu_{subject}",
                "filler_type": "short_context_mcq",
                "target_tokens": len(row["question"].split())
                + sum(len(str(choice).split()) for choice in row["choices"]),
                "position_label": "natural",
                "target_position": None,
                "actual_position": None,
                "system_prompt": SYSTEM_PROMPT,
                "prompt": prompt_for(row),
                "target": {
                    "answer": LETTERS[answer_index],
                    "answers": [LETTERS[answer_index]],
                    "evidence_ids": [],
                    "evidence_quotes": [],
                },
                "metadata": {
                    "answer_metric": "exact",
                    "evidence_id_applicable": False,
                    "evidence_quote_applicable": False,
                    "source_dataset": DATASET_ID,
                    "source_revision": DATASET_REVISION,
                    "source_split": "test",
                    "source_index": index,
                    "subject": subject,
                    "position_controlled": False,
                },
            }
        )
    sample_ids = [row["sample_id"] for row in output_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit("Duplicate MMLU sample IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    subjects = Counter(row["metadata"]["subject"] for row in output_rows)
    manifest = {
        "schema_version": "mmlu-regression-manifest-v1",
        "status": "validated",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "config": "all",
        "split": "test",
        "license": "MIT",
        "rows": len(output_rows),
        "subjects": len(subjects),
        "subject_counts": dict(sorted(subjects.items())),
        "protocol": "zero-shot generative multiple choice; exact option-letter accuracy",
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote MMLU regression set: rows={len(output_rows)} subjects={len(subjects)} "
        f"sha256={manifest['output_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
