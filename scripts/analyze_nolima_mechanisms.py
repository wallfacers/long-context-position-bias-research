#!/usr/bin/env python3
"""Analyze NoLiMa retrieval/oracle mechanisms with case-cluster bootstrap."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


RUN_ORDER = (
    "base",
    "independent_answer",
    "independent_evidence",
    "paired_answer",
    "paired_evidence",
)
RAW_METRICS = (
    "free_answer",
    "free_quote",
    "free_worst_answer",
    "locate_quote",
    "locate_worst_quote",
    "locate_supported",
    "oracle_long_answer",
    "oracle_short_answer",
)
DERIVED_METRICS = (
    "retrieval_recovery",
    "localization_gain",
    "long_distraction_penalty",
    "residual_reasoning_error",
)
METRICS = (*RAW_METRICS, *DERIVED_METRICS)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be CANONICAL_NAME=JSONL")
    name, path = value.split("=", 1)
    if name not in RUN_ORDER:
        raise argparse.ArgumentTypeError(f"Unknown canonical run: {name}")
    return name, Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * probability
    lower, upper = math.floor(coordinate), math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_p_two_sided(values: Sequence[float]) -> float:
    denominator = len(values) + 1
    return min(
        1.0,
        2
        * min(
            (sum(value <= 0 for value in values) + 1) / denominator,
            (sum(value >= 0 for value in values) + 1) / denominator,
        ),
    )


def holm_adjust(items: Sequence[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        adjusted[name] = running
    return adjusted


def add_derived(values: dict[str, float]) -> dict[str, float]:
    return values | {
        "retrieval_recovery": values["oracle_long_answer"] - values["free_answer"],
        "localization_gain": values["locate_quote"] - values["free_quote"],
        "long_distraction_penalty": values["oracle_short_answer"]
        - values["oracle_long_answer"],
        "residual_reasoning_error": 1.0 - values["oracle_short_answer"],
    }


def summarize(
    records: dict[str, list[dict[str, Any]]], selected_cases: Sequence[str]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for run in RUN_ORDER:
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records[run]:
            by_case[record["case_id"]].append(record)
        selected = [record for case in selected_cases for record in by_case[case]]
        raw = {
            metric: sum(float(record[metric]) for record in selected) / len(selected)
            for metric in RAW_METRICS
        }
        output[run] = add_derived(raw)
    return output


def case_strata(records: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    case_to_task: dict[str, str] = {}
    for record in records:
        case = str(record["case_id"])
        task = str(record["task"])
        previous = case_to_task.setdefault(case, task)
        if previous != task:
            raise ValueError(f"NoLiMa case {case!r} occurs in multiple task strata")
    strata: dict[str, list[str]] = defaultdict(list)
    for case, task in sorted(case_to_task.items()):
        strata[task].append(case)
    return dict(sorted(strata.items()))


def group_records(
    source_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    free_results: dict[str, list[dict[str, Any]]],
    diagnostic_results: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    source_by_id = {row["sample_id"]: row for row in source_rows}
    diagnostic_by_id = {row["sample_id"]: row for row in diagnostic_rows}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        groups[row["group_id"]].append(row)
    if len(source_by_id) != 1050 or len(groups) != 150:
        raise ValueError("Expected 1,050 source samples in 150 position groups")
    output: dict[str, list[dict[str, Any]]] = {}
    for run in RUN_ORDER:
        free = {row["sample_id"]: row for row in free_results[run]}
        diagnostic = {row["sample_id"]: row for row in diagnostic_results[run]}
        if set(free) != set(source_by_id):
            raise ValueError(f"{run}: free result IDs differ from frozen NoLiMa data")
        if set(diagnostic) != set(diagnostic_by_id):
            raise ValueError(f"{run}: diagnostic result IDs differ from frozen data")
        records = []
        for group_id, members in sorted(groups.items()):
            members = sorted(members, key=lambda row: row["position_label"])
            case_id = str(members[0]["metadata"]["case_id"])
            free_rows = [free[row["sample_id"]] for row in members]
            locate_rows = [
                diagnostic[f"{row['sample_id']}@diag-locate_only"] for row in members
            ]
            oracle_long = diagnostic[f"{group_id}@diag-oracle_long"]
            oracle_short = diagnostic[f"{group_id}@diag-oracle_short"]
            for row in (*free_rows, *locate_rows, oracle_long, oracle_short):
                if not row.get("valid_json"):
                    # Invalid JSON remains a scored failure; this condition only
                    # documents that it was intentionally included.
                    continue
            records.append(
                {
                    "group_id": group_id,
                    "case_id": case_id,
                    "task": members[0]["task"],
                    "book": members[0]["metadata"]["book"],
                    "target_tokens": members[0]["target_tokens"],
                    "free_answer": sum(bool(row["answer_correct"]) for row in free_rows)
                    / 7,
                    "free_quote": sum(bool(row["evidence_quotes_correct"]) for row in free_rows)
                    / 7,
                    "free_worst_answer": min(bool(row["answer_correct"]) for row in free_rows),
                    "locate_quote": sum(
                        bool(row["evidence_quotes_correct"]) for row in locate_rows
                    )
                    / 7,
                    "locate_worst_quote": min(
                        bool(row["evidence_quotes_correct"]) for row in locate_rows
                    ),
                    "locate_supported": sum(
                        bool(row["all_predicted_quotes_supported"]) for row in locate_rows
                    )
                    / 7,
                    "oracle_long_answer": bool(oracle_long["answer_correct"]),
                    "oracle_short_answer": bool(oracle_short["answer_correct"]),
                }
            )
        output[run] = records
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--diagnostic-data", type=Path, required=True)
    parser.add_argument("--free-run", action="append", type=parse_run, required=True)
    parser.add_argument("--diagnostic-run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    free_paths = dict(args.free_run)
    diagnostic_paths = dict(args.diagnostic_run)
    if (
        set(free_paths) != set(RUN_ORDER)
        or set(diagnostic_paths) != set(RUN_ORDER)
        or len(args.free_run) != len(RUN_ORDER)
        or len(args.diagnostic_run) != len(RUN_ORDER)
    ):
        raise SystemExit("Exactly one free and diagnostic result is required per canonical run")
    if args.bootstrap_replicates < 100 or args.progress_every < 1:
        raise SystemExit("Use at least 100 bootstrap replicates")
    source_rows = read_jsonl(args.source_data)
    diagnostic_rows = read_jsonl(args.diagnostic_data)
    free_results = {run: read_jsonl(free_paths[run]) for run in RUN_ORDER}
    diagnostic_results = {
        run: read_jsonl(diagnostic_paths[run]) for run in RUN_ORDER
    }
    records = group_records(
        source_rows, diagnostic_rows, free_results, diagnostic_results
    )
    strata = case_strata(records["base"])
    cases = sorted(case for stratum_cases in strata.values() for case in stratum_cases)
    if len(cases) != 10 or any(len(records[run]) != 150 for run in RUN_ORDER):
        raise SystemExit("Expected 10 case clusters and 150 group records per run")
    point = summarize(records, cases)
    run_samples = {(run, metric): [] for run in RUN_ORDER for metric in METRICS}
    effect_samples = {
        (run, metric): []
        for run in RUN_ORDER
        if run != "base"
        for metric in METRICS
    }
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices_path = args.output_dir / "case_cluster_bootstrap_indices.jsonl.gz"
    bootstrap_started = time.monotonic()
    with gzip.open(indices_path, "wt", encoding="utf-8") as handle:
        for replicate in range(args.bootstrap_replicates):
            indices_by_task = {}
            selected = []
            for task, task_cases in strata.items():
                indices = [rng.randrange(len(task_cases)) for _ in task_cases]
                indices_by_task[task] = indices
                selected.extend(task_cases[index] for index in indices)
            values = summarize(records, selected)
            handle.write(
                json.dumps(
                    {
                        "replicate": replicate,
                        "case_indices_by_task": indices_by_task,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            for run in RUN_ORDER:
                for metric in METRICS:
                    run_samples[(run, metric)].append(values[run][metric])
                    if run != "base":
                        effect_samples[(run, metric)].append(
                            values[run][metric] - values["base"][metric]
                        )
            completed = replicate + 1
            if completed % args.progress_every == 0 or completed == args.bootstrap_replicates:
                elapsed = time.monotonic() - bootstrap_started
                remaining = args.bootstrap_replicates - completed
                print(
                    f"bootstrap progress={completed}/{args.bootstrap_replicates} "
                    f"elapsed={elapsed:.1f}s eta={elapsed / completed * remaining:.1f}s",
                    flush=True,
                )
    intervals = {
        run: {
            metric: {
                "estimate": point[run][metric],
                "ci95_low": percentile(run_samples[(run, metric)], 0.025),
                "ci95_high": percentile(run_samples[(run, metric)], 0.975),
            }
            for metric in METRICS
        }
        for run in RUN_ORDER
    }
    effects: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        raw_p = []
        for run in RUN_ORDER:
            if run == "base":
                continue
            samples = effect_samples[(run, metric)]
            p_value = bootstrap_p_two_sided(samples)
            raw_p.append((run, p_value))
            effects.setdefault(run, {})[metric] = {
                "estimate": point[run][metric] - point["base"][metric],
                "ci95_low": percentile(samples, 0.025),
                "ci95_high": percentile(samples, 0.975),
                "bootstrap_p_two_sided": p_value,
            }
        for run, adjusted in holm_adjust(raw_p).items():
            effects[run][metric]["holm_adjusted_p"] = adjusted
    diagnostics = {}
    for run, rows in diagnostic_results.items():
        diagnostics[run] = {
            "finish_reason_counts": dict(
                sorted(Counter(str(row.get("finish_reason", "unknown")) for row in rows).items())
            ),
            "invalid_json": sum(not row.get("valid_json") for row in rows),
        }
    report = {
        "schema_version": "nolima-mechanism-analysis-v1",
        "runs": list(RUN_ORDER),
        "source_rows": len(source_rows),
        "diagnostic_rows_per_run": len(diagnostic_rows),
        "groups": 150,
        "case_clusters": len(cases),
        "metrics": list(METRICS),
        "interpretation": {
            "retrieval_recovery": "oracle-long answer minus free answer; positive values indicate recoverable retrieval failure",
            "localization_gain": "locate-only quote minus free quote",
            "long_distraction_penalty": "oracle-short answer minus oracle-long answer; positive values indicate residual long-context interference",
            "residual_reasoning_error": "one minus oracle-short answer accuracy",
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "unit": "NoLiMa semantic case cluster, stratified by task and paired across runs and modes",
            "clusters_per_stratum": {
                task: len(task_cases) for task, task_cases in strata.items()
            },
            "indices": indices_path.name,
        },
        "run_intervals": intervals,
        "effects_vs_base": effects,
        "generation_diagnostics": diagnostics,
    }
    analysis_path = args.output_dir / "nolima_mechanism_analysis.json"
    analysis_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "nolima_mechanism_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "metric", "estimate", "ci95_low", "ci95_high"))
        for run in RUN_ORDER:
            for metric in METRICS:
                item = intervals[run][metric]
                writer.writerow(
                    (run, metric, item["estimate"], item["ci95_low"], item["ci95_high"])
                )
    print(f"Wrote NoLiMa mechanism analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
