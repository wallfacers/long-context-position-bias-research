#!/usr/bin/env python3
"""Validate the complete test matrix and write an execution/cost report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    "valid_json",
    "answer_correct",
    "evidence_ids_correct",
    "evidence_quotes_correct",
    "all_predicted_quotes_supported",
)
REQUIRED_KEYS = {
    "run_name",
    "sample_id",
    "task",
    "filler_type",
    "target_tokens",
    "position_label",
    "batch_wall_seconds",
    "finish_reason",
    "output_tokens",
} | set(METRICS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--results-dir", type=Path, required=True)
    result.add_argument("--run", action="append", required=True)
    result.add_argument("--expected-per-run", type=int, default=4200)
    result.add_argument("--expected-cells", type=int, default=84)
    result.add_argument("--expected-per-cell", type=int, default=50)
    result.add_argument("--wall-seconds", type=float, required=True)
    result.add_argument("--hourly-rate-cny", type=float, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    reference_ids: set[str] | None = None
    reference_cells: Counter[tuple[Any, ...]] | None = None
    reports: list[dict[str, Any]] = []
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
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise SystemExit(f"{run_name}: sample IDs differ from other runs")
        for row in rows:
            missing = REQUIRED_KEYS - row.keys()
            if missing:
                raise SystemExit(
                    f"{run_name}/{row.get('sample_id')}: missing {sorted(missing)}"
                )
            if row["run_name"] != run_name:
                raise SystemExit(f"{run_name}: wrong run_name in result row")
        cells = Counter(
            (
                row["task"],
                row["filler_type"],
                int(row["target_tokens"]),
                row["position_label"],
            )
            for row in rows
        )
        if len(cells) != args.expected_cells or set(cells.values()) != {
            args.expected_per_cell
        }:
            raise SystemExit(
                f"{run_name}: expected {args.expected_cells} cells x "
                f"{args.expected_per_cell}, got cells={len(cells)} counts={dict(Counter(cells.values()))}"
            )
        if reference_cells is None:
            reference_cells = cells
        elif cells != reference_cells:
            raise SystemExit(f"{run_name}: condition-cell counts differ from other runs")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "selection_complete":
            raise SystemExit(f"{run_name}: metadata not selection_complete")
        if int(metadata.get("selected_samples", -1)) != args.expected_per_run:
            raise SystemExit(f"{run_name}: metadata selected_samples mismatch")
        batch_size = int(metadata["batch_size"])
        inference_seconds = sum(float(row["batch_wall_seconds"]) for row in rows) / batch_size
        reports.append(
            {
                "run_name": run_name,
                "samples": len(rows),
                "inference_seconds_estimate": inference_seconds,
                "seconds_per_sample": inference_seconds / len(rows),
                "finish_reason_counts": dict(
                    sorted(Counter(str(row["finish_reason"]) for row in rows).items())
                ),
                "finish_reason_length_rate": sum(
                    row["finish_reason"] == "length" for row in rows
                )
                / len(rows),
                "mean_output_tokens": sum(int(row["output_tokens"]) for row in rows)
                / len(rows),
                "max_output_tokens": max(int(row["output_tokens"]) for row in rows),
            }
            | {
                metric: sum(bool(row[metric]) for row in rows) / len(rows)
                for metric in METRICS
            }
        )

    expected_cost = args.wall_seconds / 3600 * args.hourly_rate_cny
    payload = {
        "schema_version": "position-eval-full-validation-v1",
        "status": "validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": reports,
        "matrix": {
            "runs": len(args.run),
            "samples_per_run": args.expected_per_run,
            "total_samples": args.expected_per_run * len(args.run),
            "cells_per_run": args.expected_cells,
            "samples_per_cell": args.expected_per_cell,
        },
        "execution": {
            "wall_seconds": args.wall_seconds,
            "wall_hours": args.wall_seconds / 3600,
            "hourly_rate_cny": args.hourly_rate_cny,
            "estimated_cost_cny": expected_cost,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("FULL EVALUATION VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
