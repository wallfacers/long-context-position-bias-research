#!/usr/bin/env python3
"""Validate an evaluation gate and estimate full-suite time and GPU cost."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_RESULT_KEYS = {
    "schema_version",
    "run_name",
    "sample_id",
    "group_id",
    "task",
    "filler_type",
    "target_tokens",
    "position_label",
    "answer_correct",
    "evidence_ids_correct",
    "valid_json",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def token_total(path: Path) -> int:
    total = 0
    for row in read_jsonl(path):
        # Match the target-token field persisted in evaluation results so the
        # numerator and denominator use the same length definition.
        total += int(row["target_tokens"])
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--expected-per-run", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--hourly-rate-cny", type=float, required=True)
    parser.add_argument("--contingency-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.expected_per_run <= 0 or args.wall_seconds <= 0:
        raise SystemExit("Expected samples and wall time must be positive")

    sample_ids: set[str] | None = None
    selected_input_tokens = 0
    inference_seconds = 0.0
    run_reports: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for run_name in args.run:
        result_path = args.results_dir / f"{run_name}.jsonl"
        metadata_path = args.results_dir / f"{run_name}.jsonl.run.json"
        rows = read_jsonl(result_path)
        if len(rows) != args.expected_per_run:
            raise SystemExit(
                f"{run_name}: expected {args.expected_per_run} rows, found {len(rows)}"
            )
        ids = {str(row.get("sample_id")) for row in rows}
        if len(ids) != len(rows):
            raise SystemExit(f"{run_name}: duplicate sample IDs")
        if sample_ids is None:
            sample_ids = ids
            selected_input_tokens = sum(int(row["target_tokens"]) for row in rows)
        elif ids != sample_ids:
            raise SystemExit(f"{run_name}: sample selection differs from the other runs")
        for row in rows:
            missing = REQUIRED_RESULT_KEYS - row.keys()
            if missing:
                raise SystemExit(f"{run_name}/{row.get('sample_id')}: missing {sorted(missing)}")
            if row["run_name"] != run_name:
                raise SystemExit(f"{run_name}: result contains wrong run_name")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "selection_complete":
            raise SystemExit(f"{run_name}: metadata is not selection_complete")
        if int(metadata.get("selected_samples", -1)) != args.expected_per_run:
            raise SystemExit(f"{run_name}: metadata selected_samples mismatch")
        elapsed = float(metadata["elapsed_seconds_this_invocation"])
        inference_seconds += elapsed
        all_rows.extend(rows)
        run_reports.append(
            {
                "run_name": run_name,
                "samples": len(rows),
                "elapsed_seconds": elapsed,
                "seconds_per_sample": elapsed / len(rows),
                "valid_json": sum(bool(row["valid_json"]) for row in rows) / len(rows),
                "answer_accuracy": sum(bool(row["answer_correct"]) for row in rows) / len(rows),
                "evidence_id_accuracy": sum(bool(row["evidence_ids_correct"]) for row in rows)
                / len(rows),
            }
        )

    if sample_ids is None:
        raise SystemExit("No runs were validated")
    tasks = Counter(str(row["task"]) for row in all_rows[: args.expected_per_run])
    positions = Counter(str(row["position_label"]) for row in all_rows[: args.expected_per_run])
    if len(tasks) < 2 or len(positions) < 7:
        raise SystemExit(
            f"Gate coverage is insufficient: tasks={dict(tasks)}, positions={dict(positions)}"
        )

    full_input_tokens_per_run = token_total(args.test_data)
    input_token_ratio = full_input_tokens_per_run / selected_input_tokens
    fixed_seconds = max(0.0, args.wall_seconds - inference_seconds)
    projected_seconds = fixed_seconds + inference_seconds * input_token_ratio
    expected_cost = projected_seconds / 3600 * args.hourly_rate_cny
    ceiling_cost = expected_cost * (1 + args.contingency_ratio)
    report = {
        "schema_version": "position-eval-gate-v1",
        "status": "validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": run_reports,
        "gate": {
            "runs": len(args.run),
            "samples_per_run": args.expected_per_run,
            "total_samples": len(all_rows),
            "task_coverage": dict(sorted(tasks.items())),
            "position_coverage": dict(sorted(positions.items())),
            "wall_seconds": args.wall_seconds,
            "inference_seconds": inference_seconds,
        },
        "full_test_projection": {
            "method": "linear scaling by total input tokens; use as a calibration estimate",
            "test_samples_per_run": sum(1 for _ in args.test_data.open(encoding="utf-8")),
            "total_samples": sum(1 for _ in args.test_data.open(encoding="utf-8")) * len(args.run),
            "selected_input_tokens_per_run": selected_input_tokens,
            "test_input_tokens_per_run": full_input_tokens_per_run,
            "input_token_ratio": input_token_ratio,
            "projected_hours": projected_seconds / 3600,
            "hourly_rate_cny": args.hourly_rate_cny,
            "expected_cost_cny": expected_cost,
            "contingency_ratio": args.contingency_ratio,
            "budget_ceiling_cny": ceiling_cost,
            "suggested_round_up_cny": math.ceil(ceiling_cost / 10) * 10,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("EVALUATION GATE VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
