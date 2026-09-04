#!/usr/bin/env python3
"""Aggregate exact-match metrics, position gaps, and group consistency."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "valid_json",
    "answer_correct",
    "evidence_ids_correct",
    "evidence_quotes_correct",
    "all_predicted_quotes_supported",
)

APPLICABILITY_KEYS = {
    "evidence_ids_correct": "evidence_ids_applicable",
    "evidence_quotes_correct": "evidence_quotes_applicable",
    "all_predicted_quotes_supported": "all_predicted_quotes_supported_applicable",
}


def read_results(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["run_name"], row["sample_id"])
                if key in keys:
                    raise ValueError(f"Duplicate result {key} at {path}:{line_number}")
                keys.add(key)
                rows.append(row)
    return rows


def rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"n": len(rows)}
    for metric in METRICS:
        applicable_key = APPLICABILITY_KEYS.get(metric)
        applicable = [
            row
            for row in rows
            if applicable_key is None or row.get(applicable_key, True)
        ]
        report[f"{metric}_n"] = len(applicable)
        report[metric] = (
            sum(bool(row[metric]) for row in applicable) / len(applicable)
            if applicable
            else None
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_results(args.paths)
    if not rows:
        raise SystemExit("No results")

    cell_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["run_name"],
            row.get("evaluation_mode", "free"),
            row["task"],
            row["filler_type"],
            row["target_tokens"],
            row["position_label"],
        )
        cell_groups[key].append(row)
    cells = []
    for key, group in sorted(cell_groups.items()):
        run_name, evaluation_mode, task, filler, target_tokens, position = key
        cells.append(
            {
                "run_name": run_name,
                "evaluation_mode": evaluation_mode,
                "task": task,
                "filler_type": filler,
                "target_tokens": target_tokens,
                "position_label": position,
            }
            | rates(group)
        )

    condition_cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        condition_cells[
            (
                cell["run_name"],
                cell["evaluation_mode"],
                cell["task"],
                cell["filler_type"],
                cell["target_tokens"],
            )
        ].append(cell)
    position_gaps = []
    for key, condition in sorted(condition_cells.items()):
        accuracies = [cell["answer_correct"] for cell in condition]
        position_gaps.append(
            {
                "run_name": key[0],
                "evaluation_mode": key[1],
                "task": key[2],
                "filler_type": key[3],
                "target_tokens": key[4],
                "positions": len(condition),
                "max_minus_min_answer_accuracy": max(accuracies) - min(accuracies),
                "worst_position_accuracy": min(accuracies),
                "best_position_accuracy": max(accuracies),
            }
        )

    position_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        position_groups[
            (
                row["run_name"],
                row.get("evaluation_mode", "free"),
                row["group_id"],
            )
        ].append(row)
    consistency_by_run_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (run_name, evaluation_mode, _), group in position_groups.items():
        parsed_answers = [
            row["parsed"].get("answer") if row.get("parsed") else None for row in group
        ]
        consistency_by_run_mode[(run_name, evaluation_mode)].append(
            {
                "all_positions_correct": all(row["answer_correct"] for row in group),
                "same_answer_across_positions": (
                    all(answer is not None for answer in parsed_answers)
                    and len(set(parsed_answers)) == 1
                ),
            }
        )
    group_consistency = []
    for (run_name, evaluation_mode), groups in sorted(
        consistency_by_run_mode.items()
    ):
        group_consistency.append(
            {
                "run_name": run_name,
                "evaluation_mode": evaluation_mode,
                "groups": len(groups),
                "all_positions_correct": sum(g["all_positions_correct"] for g in groups)
                / len(groups),
                "same_answer_across_positions": sum(
                    g["same_answer_across_positions"] for g in groups
                )
                / len(groups),
            }
        )

    report = {
        "schema_version": "position-eval-summary-v1",
        "rows": len(rows),
        "overall_by_run": [
            {"run_name": run_name} | rates([row for row in rows if row["run_name"] == run_name])
            for run_name in sorted({row["run_name"] for row in rows})
        ],
        "overall_by_run_mode": [
            {"run_name": run_name, "evaluation_mode": evaluation_mode}
            | rates(
                [
                    row
                    for row in rows
                    if row["run_name"] == run_name
                    and row.get("evaluation_mode", "free") == evaluation_mode
                ]
            )
            for run_name, evaluation_mode in sorted(
                {
                    (row["run_name"], row.get("evaluation_mode", "free"))
                    for row in rows
                }
            )
        ],
        "cells": cells,
        "position_gaps": position_gaps,
        "group_consistency": group_consistency,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
