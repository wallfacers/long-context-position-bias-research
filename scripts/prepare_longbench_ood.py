#!/usr/bin/env python3
"""Prepare a frozen natural multi-document QA transfer set from LongBench v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASETS = ("hotpotqa", "2wikimqa", "musique")
PROMPT = (
    "Answer the question based on the given passages. Return the shortest supported "
    "answer in the answer field. Evidence IDs and evidence quotes are not annotated for "
    "this transfer benchmark, so return empty arrays for both.\n\n"
    "The following are the given passages.\n{context}\n\n"
    "Question: {question}"
)
SYSTEM_PROMPT = (
    "Return only one JSON object with keys answer, evidence_ids, evidence_quotes, and "
    "confidence. Do not add Markdown or explanatory text."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=200)
    args = parser.parse_args()
    if args.per_dataset <= 0:
        raise SystemExit("--per-dataset must be positive")

    output_rows: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        source = args.source_dir / f"{dataset}.jsonl"
        if not source.is_file():
            raise SystemExit(f"Missing official LongBench source: {source}")
        rows = read_jsonl(source)
        if len(rows) < args.per_dataset:
            raise SystemExit(
                f"{dataset}: requested {args.per_dataset} rows but source has {len(rows)}"
            )
        source_files[dataset] = {
            "path": str(source),
            "sha256": sha256_file(source),
            "available_rows": len(rows),
        }
        for index, row in enumerate(rows[: args.per_dataset]):
            answers = [str(value) for value in row["answers"]]
            if not answers:
                raise SystemExit(f"{dataset}/{index}: no reference answers")
            sample_id = f"longbench-v1/{dataset}/{row.get('_id', index)}"
            output_rows.append(
                {
                    "schema_version": "position-group-v1",
                    "sample_id": sample_id,
                    "group_id": sample_id,
                    "split": "ood_test",
                    "task": f"longbench_{dataset}",
                    "filler_type": "natural_multidoc",
                    "target_tokens": int(row["length"]),
                    "position_label": "natural",
                    "target_position": None,
                    "actual_position": None,
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt": PROMPT.format(context=row["context"], question=row["input"]),
                    "target": {
                        "answer": answers[0],
                        "answers": answers,
                        "evidence_ids": [],
                        "evidence_quotes": [],
                    },
                    "metadata": {
                        "answer_metric": "qa_f1_en",
                        "evidence_id_applicable": False,
                        "evidence_quote_applicable": False,
                        "source_benchmark": "LongBench-v1",
                        "source_dataset": dataset,
                        "source_id": row.get("_id", index),
                        "source_length_words": int(row["length"]),
                        "language": row.get("language", "en"),
                        "position_controlled": False,
                    },
                }
            )

    sample_ids = [row["sample_id"] for row in output_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit("Duplicate output sample IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    task_counts = Counter(row["task"] for row in output_rows)
    manifest = {
        "schema_version": "longbench-natural-ood-manifest-v1",
        "status": "validated",
        "license": "LongBench repository MIT; original component datasets retain their own terms",
        "official_repository": "https://github.com/THUDM/LongBench",
        "official_repository_revision": "2e00731f8d0bff23dc4325161044d0ed8af94c1e",
        "selection": "deterministic official order; first N rows from each complete 200-row test set",
        "position_controlled": False,
        "answer_metric": "maximum English QA token F1 over official references",
        "rows": len(output_rows),
        "task_counts": dict(sorted(task_counts.items())),
        "source_files": source_files,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote LongBench natural OOD set: rows={len(output_rows)} "
        f"tasks={dict(task_counts)} sha256={manifest['output_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
