#!/usr/bin/env python3
"""Paired, task-stratified bootstrap analysis for natural transfer benchmarks."""

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


VARIANTS = (
    "independent_answer",
    "independent_evidence_id",
    "independent_evidence",
    "paired_answer",
    "paired_evidence_id",
    "paired_evidence",
)
RUN_ORDER = ("base", *VARIANTS)
TASKS = ("longbench_hotpotqa", "longbench_2wikimqa", "longbench_musique")


def contrast_definitions() -> dict[str, dict[str, float]]:
    contrasts = {
        f"{variant}_minus_base": {variant: 1.0, "base": -1.0}
        for variant in VARIANTS
    }
    for supervision in ("answer", "evidence_id", "evidence"):
        contrasts[f"paired_minus_independent_at_{supervision}"] = {
            f"paired_{supervision}": 1.0,
            f"independent_{supervision}": -1.0,
        }
    contrasts["paired_minus_independent_main_effect"] = {
        **{f"paired_{item}": 1 / 3 for item in ("answer", "evidence_id", "evidence")},
        **{
            f"independent_{item}": -1 / 3
            for item in ("answer", "evidence_id", "evidence")
        },
    }
    for high, low in (
        ("evidence_id", "answer"),
        ("evidence", "evidence_id"),
        ("evidence", "answer"),
    ):
        contrasts[f"{high}_minus_{low}_main_effect"] = {
            f"paired_{high}": 0.5,
            f"independent_{high}": 0.5,
            f"paired_{low}": -0.5,
            f"independent_{low}": -0.5,
        }
        contrasts[f"pairing_x_{high}_vs_{low}"] = {
            f"paired_{high}": 1.0,
            f"paired_{low}": -1.0,
            f"independent_{high}": -1.0,
            f"independent_{low}": 1.0,
        }
    return contrasts


CONTRASTS = contrast_definitions()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be CANONICAL_NAME=JSONL")
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
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_p_two_sided(values: Sequence[float]) -> float:
    denominator = len(values) + 1
    lower = (sum(value <= 0 for value in values) + 1) / denominator
    upper = (sum(value >= 0 for value in values) + 1) / denominator
    return min(1.0, 2 * min(lower, upper))


def holm_adjust(items: Sequence[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        adjusted[name] = running
    return adjusted


def length_bin(words: int) -> str:
    if words < 4000:
        return "short_lt4k_words"
    if words < 8000:
        return "medium_4k_8k_words"
    return "long_ge8k_words"


def weighted(values: dict[str, float], weights: dict[str, float]) -> float:
    return sum(values[run] * coefficient for run, coefficient in weights.items())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100 or args.progress_every < 1:
        raise SystemExit("Use at least 100 bootstrap replicates")
    paths = dict(args.run)
    if set(paths) != set(RUN_ORDER) or len(args.run) != len(RUN_ORDER):
        raise SystemExit("Exactly one result path is required for every canonical run")
    raw = {run: read_jsonl(paths[run]) for run in RUN_ORDER}
    counts = {run: len(rows) for run, rows in raw.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise SystemExit(f"Runs must be non-empty and equal-sized: {counts}")
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    reference_ids: set[str] | None = None
    for run, rows in raw.items():
        table = {str(row["sample_id"]): row for row in rows}
        if len(table) != len(rows):
            raise SystemExit(f"{run}: duplicate sample IDs")
        if reference_ids is None:
            reference_ids = set(table)
        elif set(table) != reference_ids:
            raise SystemExit(f"{run}: sample IDs differ from base")
        indexed[run] = table
    assert reference_ids is not None
    base = indexed["base"]
    ids_by_task: dict[str, list[str]] = defaultdict(list)
    ids_by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sample_id in sorted(reference_ids):
        task = str(base[sample_id]["task"])
        if task not in TASKS:
            raise SystemExit(f"Unexpected transfer task: {task}")
        ids_by_task[task].append(sample_id)
        ids_by_stratum[
            (task, length_bin(int(base[sample_id]["target_tokens"])))
        ].append(sample_id)
        for run in RUN_ORDER:
            score = indexed[run][sample_id].get("answer_score")
            if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                raise SystemExit(f"{run}/{sample_id}: invalid answer_score {score!r}")

    slices: dict[str, list[str]] = {"overall": sorted(reference_ids)}
    for task, ids in sorted(ids_by_task.items()):
        slices[task] = ids
    for name in ("short_lt4k_words", "medium_4k_8k_words", "long_ge8k_words"):
        selected = [
            sample_id
            for sample_id in sorted(reference_ids)
            if length_bin(int(base[sample_id]["target_tokens"])) == name
        ]
        if selected:
            slices[name] = selected

    def summarize(selection: dict[str, list[str]]) -> dict[str, float]:
        chosen = [sample_id for task in sorted(selection) for sample_id in selection[task]]
        return {
            run: sum(float(indexed[run][sample_id]["answer_score"]) for sample_id in chosen)
            / len(chosen)
            for run in RUN_ORDER
        }

    point_by_slice: dict[str, dict[str, float]] = {}
    for name, sample_ids in slices.items():
        selection = defaultdict(list)
        for sample_id in sample_ids:
            selection[str(base[sample_id]["task"])].append(sample_id)
        point_by_slice[name] = summarize(dict(selection))

    run_samples = {
        (slice_name, run): []
        for slice_name in slices
        for run in RUN_ORDER
    }
    contrast_samples = {
        (slice_name, contrast): []
        for slice_name in slices
        for contrast in CONTRASTS
    }
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices_path = args.output_dir / "paired_bootstrap_indices.jsonl.gz"
    bootstrap_started = time.monotonic()
    with gzip.open(indices_path, "wt", encoding="utf-8") as handle:
        for replicate in range(args.bootstrap_replicates):
            resampled = {
                stratum: [ids[rng.randrange(len(ids))] for _ in ids]
                for stratum, ids in sorted(ids_by_stratum.items())
            }
            handle.write(
                json.dumps(
                    {
                        "replicate": replicate,
                        "task_length_indices": {
                            "|".join(stratum): [
                                ids_by_stratum[stratum].index(value) for value in values
                            ]
                            for stratum, values in resampled.items()
                        },
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            for slice_name, slice_ids in slices.items():
                allowed = set(slice_ids)
                selection = {
                    "|".join(stratum): [
                        sample_id for sample_id in values if sample_id in allowed
                    ]
                    for stratum, values in resampled.items()
                }
                selection = {task: values for task, values in selection.items() if values}
                estimates = summarize(selection)
                for run, value in estimates.items():
                    run_samples[(slice_name, run)].append(value)
                for contrast, weights in CONTRASTS.items():
                    contrast_samples[(slice_name, contrast)].append(
                        weighted(estimates, weights)
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

    intervals: dict[str, dict[str, Any]] = {}
    for slice_name, estimates in point_by_slice.items():
        intervals[slice_name] = {
            run: {
                "estimate": estimate,
                "ci95_low": percentile(run_samples[(slice_name, run)], 0.025),
                "ci95_high": percentile(run_samples[(slice_name, run)], 0.975),
                "n": len(slices[slice_name]),
            }
            for run, estimate in estimates.items()
        }

    contrasts: dict[str, dict[str, Any]] = {}
    for slice_name, estimates in point_by_slice.items():
        raw_p = []
        contrasts[slice_name] = {}
        for name, weights in CONTRASTS.items():
            samples = contrast_samples[(slice_name, name)]
            p_value = bootstrap_p_two_sided(samples)
            raw_p.append((name, p_value))
            contrasts[slice_name][name] = {
                "estimate": weighted(estimates, weights),
                "ci95_low": percentile(samples, 0.025),
                "ci95_high": percentile(samples, 0.975),
                "bootstrap_p_two_sided": p_value,
                "weights": weights,
            }
        adjusted = holm_adjust(raw_p)
        for name, value in adjusted.items():
            contrasts[slice_name][name]["holm_adjusted_p"] = value

    trained = [point_by_slice["overall"][run] for run in VARIANTS]
    report = {
        "schema_version": "natural-transfer-analysis-v1",
        "rows_per_run": counts,
        "tasks": {task: len(ids) for task, ids in sorted(ids_by_task.items())},
        "metric": "maximum LongBench English QA token F1 over references",
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "unit": (
                "natural benchmark question, paired across runs and stratified by "
                "task x official word-length bin"
            ),
            "indices": indices_path.name,
        },
        "run_intervals": intervals,
        "contrasts": contrasts,
        "benchmark_discrimination": {
            "trained_score_min": min(trained),
            "trained_score_max": max(trained),
            "trained_score_range": max(trained) - min(trained),
            "saturated_at_95pct_and_lt_2pp_range": min(trained) >= 0.95
            and max(trained) - min(trained) < 0.02,
        },
        "generation_diagnostics": {
            run: {
                "valid_json_rate": sum(bool(row["valid_json"]) for row in rows) / len(rows),
                "finish_reason_counts": dict(
                    sorted(Counter(str(row.get("finish_reason", "unknown")) for row in rows).items())
                ),
                "mean_output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows)
                / len(rows),
            }
            for run, rows in raw.items()
        },
    }
    (args.output_dir / "transfer_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_rows = [
        {
            "slice": slice_name,
            "run_name": run,
            **intervals[slice_name][run],
        }
        for slice_name in slices
        for run in RUN_ORDER
    ]
    write_csv(
        args.output_dir / "transfer_summary.csv",
        summary_rows,
        ("slice", "run_name", "estimate", "ci95_low", "ci95_high", "n"),
    )
    contrast_rows = [
        {"slice": slice_name, "contrast": name, **payload}
        for slice_name, items in contrasts.items()
        for name, payload in items.items()
    ]
    for row in contrast_rows:
        row.pop("weights")
    write_csv(
        args.output_dir / "transfer_contrasts.csv",
        contrast_rows,
        (
            "slice",
            "contrast",
            "estimate",
            "ci95_low",
            "ci95_high",
            "bootstrap_p_two_sided",
            "holm_adjusted_p",
        ),
    )
    print(f"Wrote paired natural-transfer analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
