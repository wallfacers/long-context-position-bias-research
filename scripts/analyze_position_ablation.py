#!/usr/bin/env python3
"""Paired, stratified bootstrap analysis for the 2x2 position-bias ablation."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


RUN_FACTORS: dict[str, dict[str, str | None]] = {
    "base": {"pairing": None, "supervision": None},
    "paired_evidence": {"pairing": "paired", "supervision": "evidence"},
    "paired_answer": {"pairing": "paired", "supervision": "answer"},
    "independent_evidence": {
        "pairing": "independent",
        "supervision": "evidence",
    },
    "independent_answer": {
        "pairing": "independent",
        "supervision": "answer",
    },
}
RUN_ORDER = tuple(RUN_FACTORS)
POSITIONS = ("p000", "p010", "p025", "p050", "p075", "p090", "p100")
METRICS = (
    "valid_json",
    "answer_correct",
    "evidence_ids_correct",
    "evidence_quotes_correct",
    "all_predicted_quotes_supported",
)
SUMMARY_STATS = (
    *METRICS,
    "mean_worst_position_accuracy",
    "mean_position_gap",
    "mean_edge_accuracy",
    "mean_middle_accuracy",
    "mean_middle_penalty",
)
CONTRAST_WEIGHTS: dict[str, dict[str, float]] = {
    **{
        f"{run_name}_minus_base": {run_name: 1.0, "base": -1.0}
        for run_name in RUN_ORDER
        if run_name != "base"
    },
    "paired_minus_independent_main_effect": {
        "paired_evidence": 0.5,
        "paired_answer": 0.5,
        "independent_evidence": -0.5,
        "independent_answer": -0.5,
    },
    "evidence_minus_answer_main_effect": {
        "paired_evidence": 0.5,
        "independent_evidence": 0.5,
        "paired_answer": -0.5,
        "independent_answer": -0.5,
    },
    "pairing_x_supervision_interaction": {
        "paired_evidence": 1.0,
        "paired_answer": -1.0,
        "independent_evidence": -1.0,
        "independent_answer": 1.0,
    },
}


Condition = tuple[str, str, int]
Profile = tuple[str, int, str]


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


def condition_of(row: dict[str, Any]) -> Condition:
    return row["task"], row["filler_type"], int(row["target_tokens"])


def profile_of(row: dict[str, Any]) -> Profile:
    return row["task"], int(row["target_tokens"]), row["position_label"]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take percentile of an empty list")
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_p_two_sided(values: list[float]) -> float:
    denominator = len(values) + 1
    nonpositive = (sum(value <= 0 for value in values) + 1) / denominator
    nonnegative = (sum(value >= 0 for value in values) + 1) / denominator
    return min(1.0, 2 * min(nonpositive, nonnegative))


def holm_adjust(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * p_value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def weighted_contrast(
    summaries: dict[str, dict[str, float]], weights: dict[str, float], statistic: str
) -> float:
    return sum(weight * summaries[run_name][statistic] for run_name, weight in weights.items())


def build_group_tables(
    rows_by_run: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[Condition, list[str]],
]:
    groups_by_run: dict[str, dict[str, list[dict[str, Any]]]] = {}
    condition_groups: dict[Condition, list[str]] = defaultdict(list)
    reference_ids: set[str] | None = None
    for run_name in RUN_ORDER:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen_samples: set[str] = set()
        for row in rows_by_run[run_name]:
            if row["run_name"] != run_name:
                raise SystemExit(f"{run_name}: row has run_name={row['run_name']}")
            sample_id = str(row["sample_id"])
            if sample_id in seen_samples:
                raise SystemExit(f"{run_name}: duplicate sample_id {sample_id}")
            seen_samples.add(sample_id)
            grouped[str(row["group_id"])].append(row)
        if reference_ids is None:
            reference_ids = seen_samples
        elif seen_samples != reference_ids:
            raise SystemExit(f"{run_name}: sample IDs differ from the other runs")
        for group_id, group in grouped.items():
            conditions = {condition_of(row) for row in group}
            positions = [row["position_label"] for row in group]
            if len(conditions) != 1 or set(positions) != set(POSITIONS) or len(positions) != 7:
                raise SystemExit(
                    f"{run_name}/{group_id}: expected one condition and seven unique positions"
                )
        groups_by_run[run_name] = dict(grouped)
        if run_name == "base":
            for group_id, group in grouped.items():
                condition_groups[condition_of(group[0])].append(group_id)
    for condition in condition_groups:
        condition_groups[condition].sort()
    return groups_by_run, dict(condition_groups)


def summarize_selection(
    grouped: dict[str, list[dict[str, Any]]],
    selected: dict[Condition, list[str]],
) -> tuple[dict[str, float], dict[Profile, float]]:
    metric_sums = Counter()
    total_rows = 0
    position_sums: dict[Condition, Counter[str]] = defaultdict(Counter)
    position_counts: dict[Condition, Counter[str]] = defaultdict(Counter)
    profile_sums: Counter[Profile] = Counter()
    profile_counts: Counter[Profile] = Counter()
    for condition, group_ids in selected.items():
        for group_id in group_ids:
            for row in grouped[group_id]:
                total_rows += 1
                for metric in METRICS:
                    metric_sums[metric] += int(bool(row[metric]))
                position = row["position_label"]
                position_sums[condition][position] += int(bool(row["answer_correct"]))
                position_counts[condition][position] += 1
                profile = profile_of(row)
                profile_sums[profile] += int(bool(row["answer_correct"]))
                profile_counts[profile] += 1
    if total_rows == 0:
        raise ValueError("Empty selection")
    summary = {metric: metric_sums[metric] / total_rows for metric in METRICS}
    worst: list[float] = []
    gaps: list[float] = []
    edges: list[float] = []
    middles: list[float] = []
    penalties: list[float] = []
    for condition in sorted(selected):
        rates = {
            position: position_sums[condition][position]
            / position_counts[condition][position]
            for position in POSITIONS
        }
        worst.append(min(rates.values()))
        gaps.append(max(rates.values()) - min(rates.values()))
        edge = (rates["p000"] + rates["p100"]) / 2
        middle = rates["p050"]
        edges.append(edge)
        middles.append(middle)
        penalties.append(edge - middle)
    summary.update(
        {
            "mean_worst_position_accuracy": sum(worst) / len(worst),
            "mean_position_gap": sum(gaps) / len(gaps),
            "mean_edge_accuracy": sum(edges) / len(edges),
            "mean_middle_accuracy": sum(middles) / len(middles),
            "mean_middle_penalty": sum(penalties) / len(penalties),
        }
    )
    profiles = {
        profile: profile_sums[profile] / profile_counts[profile]
        for profile in profile_sums
    }
    return summary, profiles


def write_summary_csv(path: Path, summaries: dict[str, dict[str, float]]) -> None:
    fields = ("run_name", "pairing", "supervision", *SUMMARY_STATS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run_name in RUN_ORDER:
            writer.writerow(
                {"run_name": run_name, **RUN_FACTORS[run_name], **summaries[run_name]}
            )


def write_cells_csv(path: Path, rows_by_run: dict[str, list[dict[str, Any]]]) -> None:
    fields = (
        "run_name",
        "pairing",
        "supervision",
        "task",
        "filler_type",
        "target_tokens",
        "position_label",
        "n",
        "finish_reason_length_rate",
        "mean_output_tokens",
        *METRICS,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run_name in RUN_ORDER:
            cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in rows_by_run[run_name]:
                cells[
                    (
                        row["task"],
                        row["filler_type"],
                        int(row["target_tokens"]),
                        row["position_label"],
                    )
                ].append(row)
            for key, rows in sorted(cells.items()):
                writer.writerow(
                    {
                        "run_name": run_name,
                        **RUN_FACTORS[run_name],
                        "task": key[0],
                        "filler_type": key[1],
                        "target_tokens": key[2],
                        "position_label": key[3],
                        "n": len(rows),
                        "finish_reason_length_rate": sum(
                            row.get("finish_reason") == "length" for row in rows
                        )
                        / len(rows),
                        "mean_output_tokens": sum(
                            int(row.get("output_tokens", 0)) for row in rows
                        )
                        / len(rows),
                        **{
                            metric: sum(bool(row[metric]) for row in rows) / len(rows)
                            for metric in METRICS
                        },
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("Use at least 100 bootstrap replicates")

    rows_by_run = {
        run_name: read_jsonl(args.results_dir / f"{run_name}.jsonl")
        for run_name in RUN_ORDER
    }
    counts = {run_name: len(rows) for run_name, rows in rows_by_run.items()}
    if len(set(counts.values())) != 1 or next(iter(counts.values())) == 0:
        raise SystemExit(f"Runs must be non-empty and equal-sized: {counts}")
    grouped, condition_groups = build_group_tables(rows_by_run)
    point_summaries: dict[str, dict[str, float]] = {}
    point_profiles: dict[str, dict[Profile, float]] = {}
    for run_name in RUN_ORDER:
        point_summaries[run_name], point_profiles[run_name] = summarize_selection(
            grouped[run_name], condition_groups
        )

    contrast_samples: dict[tuple[str, str], list[float]] = {
        (contrast, statistic): []
        for contrast in CONTRAST_WEIGHTS
        for statistic in SUMMARY_STATS
    }
    summary_samples: dict[tuple[str, str], list[float]] = {
        (run_name, statistic): []
        for run_name in RUN_ORDER
        for statistic in SUMMARY_STATS
    }
    profile_samples: dict[tuple[str, Profile], list[float]] = {
        (run_name, profile): []
        for run_name in RUN_ORDER
        for profile in point_profiles[run_name]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices_path = args.output_dir / "paired_bootstrap_indices.jsonl.gz"
    rng = random.Random(args.seed)
    with gzip.open(indices_path, "wt", encoding="utf-8") as index_handle:
        for replicate in range(args.bootstrap_replicates):
            selected: dict[Condition, list[str]] = {}
            serialized_indices: dict[str, list[int]] = {}
            for condition in sorted(condition_groups):
                candidates = condition_groups[condition]
                indices = [rng.randrange(len(candidates)) for _ in candidates]
                selected[condition] = [candidates[index] for index in indices]
                serialized_indices["|".join(map(str, condition))] = indices
            index_handle.write(
                json.dumps(
                    {"replicate": replicate, "condition_indices": serialized_indices},
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            summaries: dict[str, dict[str, float]] = {}
            for run_name in RUN_ORDER:
                summaries[run_name], profiles = summarize_selection(
                    grouped[run_name], selected
                )
                for statistic in SUMMARY_STATS:
                    summary_samples[(run_name, statistic)].append(
                        summaries[run_name][statistic]
                    )
                for profile, value in profiles.items():
                    profile_samples[(run_name, profile)].append(value)
            for contrast_name, weights in CONTRAST_WEIGHTS.items():
                for statistic in SUMMARY_STATS:
                    contrast_samples[(contrast_name, statistic)].append(
                        weighted_contrast(summaries, weights, statistic)
                    )

    contrasts: dict[str, Any] = {}
    p_values_by_stat: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for contrast_name, weights in CONTRAST_WEIGHTS.items():
        statistics: dict[str, Any] = {}
        for statistic in SUMMARY_STATS:
            samples = contrast_samples[(contrast_name, statistic)]
            p_value = bootstrap_p_two_sided(samples)
            p_values_by_stat[statistic].append((contrast_name, p_value))
            statistics[statistic] = {
                "estimate": weighted_contrast(point_summaries, weights, statistic),
                "ci95_low": percentile(samples, 0.025),
                "ci95_high": percentile(samples, 0.975),
                "bootstrap_p_two_sided": p_value,
            }
        contrasts[contrast_name] = {"weights": weights, "statistics": statistics}
    for statistic, items in p_values_by_stat.items():
        adjusted = holm_adjust(items)
        for contrast_name, value in adjusted.items():
            contrasts[contrast_name]["statistics"][statistic]["holm_adjusted_p"] = value

    position_profiles = []
    for run_name in RUN_ORDER:
        for profile, estimate in sorted(point_profiles[run_name].items()):
            samples = profile_samples[(run_name, profile)]
            position_profiles.append(
                {
                    "run_name": run_name,
                    **RUN_FACTORS[run_name],
                    "task": profile[0],
                    "target_tokens": profile[1],
                    "position_label": profile[2],
                    "answer_accuracy": estimate,
                    "ci95_low": percentile(samples, 0.025),
                    "ci95_high": percentile(samples, 0.975),
                }
            )

    run_summary_intervals: dict[str, dict[str, dict[str, float]]] = {}
    for run_name in RUN_ORDER:
        run_summary_intervals[run_name] = {}
        for statistic in SUMMARY_STATS:
            samples = summary_samples[(run_name, statistic)]
            run_summary_intervals[run_name][statistic] = {
                "estimate": point_summaries[run_name][statistic],
                "ci95_low": percentile(samples, 0.025),
                "ci95_high": percentile(samples, 0.975),
            }

    base = point_summaries["base"]
    screening = []
    for run_name in RUN_ORDER[1:]:
        current = point_summaries[run_name]
        base_gap = base["mean_position_gap"]
        gap_reduction_fraction = (
            (base_gap - current["mean_position_gap"]) / base_gap if base_gap else None
        )
        checks = {
            "gap_reduction_at_least_50pct": gap_reduction_fraction is not None
            and gap_reduction_fraction >= 0.50,
            "worst_position_gain_at_least_10pp": current[
                "mean_worst_position_accuracy"
            ]
            - base["mean_worst_position_accuracy"]
            >= 0.10,
            "mean_answer_drop_no_more_than_2pp": current["answer_correct"]
            - base["answer_correct"]
            >= -0.02,
            "edge_accuracy_drop_no_more_than_2pp": current["mean_edge_accuracy"]
            - base["mean_edge_accuracy"]
            >= -0.02,
            "valid_json_at_least_99pct": current["valid_json"] >= 0.99,
        }
        screening.append(
            {
                "run_name": run_name,
                "gap_reduction_fraction": gap_reduction_fraction,
                "worst_position_gain": current["mean_worst_position_accuracy"]
                - base["mean_worst_position_accuracy"],
                "mean_answer_delta": current["answer_correct"] - base["answer_correct"],
                "edge_accuracy_delta": current["mean_edge_accuracy"]
                - base["mean_edge_accuracy"],
                "checks": checks,
                "passes_all_exploratory_checks": all(checks.values()),
            }
        )

    generation_diagnostics = {}
    for run_name in RUN_ORDER:
        rows = rows_by_run[run_name]
        finish_reasons = Counter(str(row.get("finish_reason", "unknown")) for row in rows)
        output_tokens = [int(row.get("output_tokens", 0)) for row in rows]
        generation_diagnostics[run_name] = {
            "finish_reason_counts": dict(sorted(finish_reasons.items())),
            "finish_reason_length_rate": finish_reasons.get("length", 0) / len(rows),
            "mean_output_tokens": sum(output_tokens) / len(output_tokens),
            "max_output_tokens": max(output_tokens),
        }

    report = {
        "schema_version": "position-ablation-analysis-v1",
        "scope": "single-seed exploratory pilot; not a final significance claim",
        "bootstrap": {
            "method": "paired group bootstrap, stratified by task x filler x length",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "indices_file": indices_path.name,
        },
        "rows_per_run": counts,
        "groups_per_condition": {
            "|".join(map(str, condition)): len(group_ids)
            for condition, group_ids in sorted(condition_groups.items())
        },
        "run_summaries": point_summaries,
        "run_summary_intervals": run_summary_intervals,
        "generation_diagnostics": generation_diagnostics,
        "position_profiles": position_profiles,
        "contrasts": contrasts,
        "exploratory_screening": {
            "criteria": {
                "gap_reduction_fraction": 0.50,
                "worst_position_gain": 0.10,
                "mean_answer_delta_floor": -0.02,
                "edge_accuracy_delta_floor": -0.02,
                "valid_json_floor": 0.99,
            },
            "results": screening,
        },
    }
    (args.output_dir / "ablation_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(args.output_dir / "ablation_summary.csv", point_summaries)
    write_cells_csv(args.output_dir / "position_cells.csv", rows_by_run)
    with (args.output_dir / "position_profiles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(position_profiles[0]))
        writer.writeheader()
        writer.writerows(position_profiles)
    contrast_rows = []
    for contrast_name, contrast in contrasts.items():
        for statistic, values in contrast["statistics"].items():
            contrast_rows.append(
                {"contrast": contrast_name, "statistic": statistic, **values}
            )
    with (args.output_dir / "ablation_contrasts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)
    print(f"Wrote paired ablation analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
