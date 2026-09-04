#!/usr/bin/env python3
"""Paired stratified bootstrap analysis for the matched 2x3 factorial study."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "independent_answer",
    "independent_evidence_id",
    "independent_evidence",
    "paired_answer",
    "paired_evidence_id",
    "paired_evidence",
)
RUN_ORDER = ("base", *VARIANTS)
FACTORS = {
    "base": {"pairing": None, "supervision": None},
    **{
        variant: {
            "pairing": variant.split("_", 1)[0],
            "supervision": variant.split("_", 1)[1],
        }
        for variant in VARIANTS
    },
}
POSITIONS = ("p000", "p010", "p025", "p050", "p075", "p090", "p100")
ROW_METRICS = (
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
SUMMARY_STATS = (
    *ROW_METRICS,
    "mean_worst_answer_accuracy",
    "mean_answer_position_gap",
    "mean_edge_answer_accuracy",
    "mean_middle_answer_accuracy",
    "mean_middle_answer_penalty",
    "mean_worst_quote_accuracy",
    "mean_quote_position_gap",
    "all_positions_answer_correct",
    "same_answer_across_positions",
)


def contrast_definitions() -> dict[str, dict[str, float]]:
    contrasts: dict[str, dict[str, float]] = {
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
    transitions = (
        ("evidence_id", "answer"),
        ("evidence", "evidence_id"),
        ("evidence", "answer"),
    )
    for high, low in transitions:
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
Condition = tuple[str, str, str, int]
Profile = tuple[str, int, str]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be CANONICAL_NAME=JSONL")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if name not in RUN_ORDER:
        raise argparse.ArgumentTypeError(
            f"unknown canonical run {name!r}; expected one of {', '.join(RUN_ORDER)}"
        )
    return name, Path(raw_path)


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


def attach_cluster_metadata(
    rows_by_run: dict[str, list[dict[str, Any]]], source_path: Path
) -> str:
    source_by_id: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    with source_path.open("rb") as raw_handle:
        for block in iter(lambda: raw_handle.read(1024 * 1024), b""):
            digest.update(block)
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in source_by_id:
                raise SystemExit(
                    f"Duplicate sample_id in cluster source {source_path}:{line_number}"
                )
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise SystemExit(
                    f"Missing metadata in cluster source {source_path}:{line_number}"
                )
            source_by_id[sample_id] = metadata
    source_ids = set(source_by_id)
    for run_name, rows in rows_by_run.items():
        result_ids = {str(row["sample_id"]) for row in rows}
        if result_ids != source_ids:
            raise SystemExit(
                f"{run_name}: cluster source sample IDs differ "
                f"(missing={len(source_ids - result_ids)}, extra={len(result_ids - source_ids)})"
            )
        for row in rows:
            sample_id = str(row["sample_id"])
            existing = row.get("metadata")
            if existing is not None and existing != source_by_id[sample_id]:
                raise SystemExit(
                    f"{run_name}/{sample_id}: result metadata conflicts with frozen cluster source"
                )
            row["metadata"] = source_by_id[sample_id]
    return digest.hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take percentile of an empty sequence")
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_p_two_sided(values: Sequence[float]) -> float:
    denominator = len(values) + 1
    nonpositive = (sum(value <= 0 for value in values) + 1) / denominator
    nonnegative = (sum(value >= 0 for value in values) + 1) / denominator
    return min(1.0, 2 * min(nonpositive, nonnegative))


def holm_adjust(items: Sequence[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (len(ordered) - index) * p_value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def condition_of(row: dict[str, Any]) -> Condition:
    return (
        str(row.get("evaluation_mode", "free")),
        str(row["task"]),
        str(row["filler_type"]),
        int(row["target_tokens"]),
    )


def metric_applicable(row: dict[str, Any], metric: str) -> bool:
    key = APPLICABILITY_KEYS.get(metric)
    return key is None or bool(row.get(key, True))


def nested_string(row: dict[str, Any], dotted_key: str) -> str:
    value: Any = row
    for component in dotted_key.split("."):
        if not isinstance(value, dict) or component not in value:
            raise SystemExit(f"Missing bootstrap key {dotted_key!r} in result row")
        value = value[component]
    rendered = str(value).strip()
    if not rendered:
        raise SystemExit(f"Empty bootstrap key {dotted_key!r} in result row")
    return rendered


def build_group_tables(
    rows_by_run: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[Condition, list[str]]]:
    groups_by_run: dict[str, dict[str, list[dict[str, Any]]]] = {}
    reference_samples: set[str] | None = None
    condition_groups: dict[Condition, list[str]] = defaultdict(list)
    for run_name in RUN_ORDER:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        samples: set[str] = set()
        for row in rows_by_run[run_name]:
            sample_id = str(row["sample_id"])
            if sample_id in samples:
                raise SystemExit(f"{run_name}: duplicate sample_id {sample_id}")
            samples.add(sample_id)
            grouped[str(row["group_id"])].append(row)
        if reference_samples is None:
            reference_samples = samples
        elif samples != reference_samples:
            missing = len(reference_samples - samples)
            extra = len(samples - reference_samples)
            raise SystemExit(
                f"{run_name}: sample IDs differ from base (missing={missing}, extra={extra})"
            )
        for group_id, group in grouped.items():
            positions = [row["position_label"] for row in group]
            conditions = {condition_of(row) for row in group}
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


def build_cluster_tables(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    cluster_key: str,
    strata_key: str | None,
) -> dict[str, dict[str, list[str]]]:
    clusters: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    cluster_strata: dict[str, str] = {}
    for group_id, rows in grouped.items():
        cluster_values = {nested_string(row, cluster_key) for row in rows}
        stratum_values = (
            {nested_string(row, strata_key) for row in rows}
            if strata_key is not None
            else {"all"}
        )
        if len(cluster_values) != 1 or len(stratum_values) != 1:
            raise SystemExit(
                f"{group_id}: cluster and stratum keys must be constant within a position group"
            )
        cluster = next(iter(cluster_values))
        stratum = next(iter(stratum_values))
        previous = cluster_strata.setdefault(cluster, stratum)
        if previous != stratum:
            raise SystemExit(
                f"Cluster {cluster!r} occurs in multiple bootstrap strata: {previous!r}, {stratum!r}"
            )
        clusters[stratum][cluster].append(group_id)
    result = {
        stratum: {
            cluster: sorted(group_ids)
            for cluster, group_ids in sorted(stratum_clusters.items())
        }
        for stratum, stratum_clusters in sorted(clusters.items())
    }
    if sum(len(stratum_clusters) for stratum_clusters in result.values()) < 2:
        raise SystemExit("Cluster bootstrap requires at least two distinct clusters")
    return result


def summarize_selection(
    grouped: dict[str, list[dict[str, Any]]],
    selected: dict[Condition, list[str]],
) -> tuple[dict[str, float | None], dict[Profile, float]]:
    chosen_rows: list[dict[str, Any]] = []
    chosen_groups: list[list[dict[str, Any]]] = []
    for condition, group_ids in selected.items():
        del condition
        for group_id in group_ids:
            group = grouped[group_id]
            chosen_groups.append(group)
            chosen_rows.extend(group)
    if not chosen_rows:
        raise ValueError("Empty bootstrap selection")

    summary: dict[str, float | None] = {}
    for metric in ROW_METRICS:
        applicable = [row for row in chosen_rows if metric_applicable(row, metric)]
        summary[metric] = (
            sum(bool(row[metric]) for row in applicable) / len(applicable)
            if applicable
            else None
        )

    condition_rows: dict[Condition, list[dict[str, Any]]] = defaultdict(list)
    for row in chosen_rows:
        condition_rows[condition_of(row)].append(row)
    worst_answer: list[float] = []
    answer_gaps: list[float] = []
    edge_answer: list[float] = []
    middle_answer: list[float] = []
    middle_penalty: list[float] = []
    worst_quote: list[float] = []
    quote_gaps: list[float] = []
    profile_sums: Counter[Profile] = Counter()
    profile_counts: Counter[Profile] = Counter()
    for condition, rows in condition_rows.items():
        by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_position[str(row["position_label"])].append(row)
            profile = (str(row["task"]), int(row["target_tokens"]), str(row["position_label"]))
            profile_sums[profile] += int(bool(row["answer_correct"]))
            profile_counts[profile] += 1
        answer_rates = {
            position: sum(bool(row["answer_correct"]) for row in by_position[position])
            / len(by_position[position])
            for position in POSITIONS
        }
        quote_rates: dict[str, float] = {}
        for position in POSITIONS:
            applicable_quotes = [
                row
                for row in by_position[position]
                if metric_applicable(row, "evidence_quotes_correct")
            ]
            if applicable_quotes:
                quote_rates[position] = sum(
                    bool(row["evidence_quotes_correct"]) for row in applicable_quotes
                ) / len(applicable_quotes)
        worst_answer.append(min(answer_rates.values()))
        answer_gaps.append(max(answer_rates.values()) - min(answer_rates.values()))
        edge = (answer_rates["p000"] + answer_rates["p100"]) / 2
        middle = answer_rates["p050"]
        edge_answer.append(edge)
        middle_answer.append(middle)
        middle_penalty.append(edge - middle)
        if set(quote_rates) == set(POSITIONS):
            worst_quote.append(min(quote_rates.values()))
            quote_gaps.append(max(quote_rates.values()) - min(quote_rates.values()))

    parsed_group_consistency = []
    for group in chosen_groups:
        parsed_answers = [
            row.get("parsed", {}).get("answer") if row.get("parsed") else None
            for row in group
        ]
        parsed_group_consistency.append(
            (
                all(bool(row["answer_correct"]) for row in group),
                all(answer is not None for answer in parsed_answers)
                and len(set(parsed_answers)) == 1,
            )
        )
    summary.update(
        {
            "mean_worst_answer_accuracy": sum(worst_answer) / len(worst_answer),
            "mean_answer_position_gap": sum(answer_gaps) / len(answer_gaps),
            "mean_edge_answer_accuracy": sum(edge_answer) / len(edge_answer),
            "mean_middle_answer_accuracy": sum(middle_answer) / len(middle_answer),
            "mean_middle_answer_penalty": sum(middle_penalty) / len(middle_penalty),
            "mean_worst_quote_accuracy": (
                sum(worst_quote) / len(worst_quote) if worst_quote else None
            ),
            "mean_quote_position_gap": (
                sum(quote_gaps) / len(quote_gaps) if quote_gaps else None
            ),
            "all_positions_answer_correct": sum(item[0] for item in parsed_group_consistency)
            / len(parsed_group_consistency),
            "same_answer_across_positions": sum(item[1] for item in parsed_group_consistency)
            / len(parsed_group_consistency),
        }
    )
    profiles = {
        profile: profile_sums[profile] / profile_counts[profile]
        for profile in profile_sums
    }
    return summary, profiles


def weighted_contrast(
    summaries: dict[str, dict[str, float | None]],
    weights: dict[str, float],
    statistic: str,
) -> float | None:
    values = [summaries[run][statistic] for run in weights]
    if any(value is None for value in values):
        return None
    return sum(weights[run] * float(summaries[run][statistic]) for run in weights)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--cluster-key",
        help=(
            "Dotted row key for cluster bootstrap. NoLiMa rows automatically use "
            "metadata.case_id when this option is omitted."
        ),
    )
    parser.add_argument(
        "--cluster-source-data",
        type=Path,
        help=(
            "Frozen input JSONL joined by sample_id when result rows omit cluster metadata. "
            "NoLiMa automatically discovers data/ood_nolima/hard_gate.jsonl."
        ),
    )
    parser.add_argument(
        "--cluster-strata-key",
        help="Optional dotted row key used to stratify cluster resampling",
    )
    parser.add_argument(
        "--expected-clusters",
        type=int,
        help="Fail unless cluster bootstrap discovers exactly this many clusters",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Emit elapsed/ETA diagnostics every N bootstrap replicates",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("Use at least 100 bootstrap replicates")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")
    if args.cluster_strata_key and not args.cluster_key:
        raise SystemExit("--cluster-strata-key requires --cluster-key")
    if args.expected_clusters is not None and args.expected_clusters < 2:
        raise SystemExit("--expected-clusters must be at least 2")
    run_paths = dict(args.run)
    if set(run_paths) != set(RUN_ORDER) or len(args.run) != len(RUN_ORDER):
        raise SystemExit(f"Exactly one path is required for each run: {', '.join(RUN_ORDER)}")
    rows_by_run = {name: read_jsonl(run_paths[name]) for name in RUN_ORDER}
    counts = {name: len(rows) for name, rows in rows_by_run.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise SystemExit(f"Runs must be non-empty and equal-sized: {counts}")
    looks_like_nolima = all(
        str(row.get("task", "")).startswith("nolima_")
        and str(row.get("group_id", "")).startswith("nolima-")
        for row in rows_by_run["base"]
    )
    cluster_source_data = args.cluster_source_data
    if cluster_source_data is None and looks_like_nolima:
        candidate = ROOT / "data/ood_nolima/hard_gate.jsonl"
        if candidate.is_file():
            cluster_source_data = candidate
    cluster_source_sha256 = None
    if cluster_source_data is not None:
        if not cluster_source_data.is_file():
            raise SystemExit(f"Cluster source data is missing: {cluster_source_data}")
        cluster_source_sha256 = attach_cluster_metadata(
            rows_by_run, cluster_source_data
        )
    grouped, condition_groups = build_group_tables(rows_by_run)
    cluster_key = args.cluster_key
    cluster_strata_key = args.cluster_strata_key
    auto_nolima = all(
        row.get("metadata", {}).get("benchmark") == "NoLiMa"
        and row.get("metadata", {}).get("case_id")
        for row in rows_by_run["base"]
    )
    if cluster_key is None and auto_nolima:
        cluster_key = "metadata.case_id"
        cluster_strata_key = "task"
        print(
            "Detected NoLiMa rows; using semantic-case cluster bootstrap stratified by task",
            flush=True,
        )
    effective_expected_clusters = args.expected_clusters
    nolima_splits = {
        str(row.get("metadata", {}).get("benchmark_split", ""))
        for row in rows_by_run["base"]
    }
    if (
        effective_expected_clusters is None
        and auto_nolima
        and len(rows_by_run["base"]) == 1050
        and nolima_splits == {"needle_set_hard"}
    ):
        effective_expected_clusters = 10
    cluster_tables = (
        build_cluster_tables(
            grouped["base"],
            cluster_key=cluster_key,
            strata_key=cluster_strata_key,
        )
        if cluster_key is not None
        else None
    )
    cluster_count = (
        sum(len(stratum_clusters) for stratum_clusters in cluster_tables.values())
        if cluster_tables is not None
        else None
    )
    if (
        effective_expected_clusters is not None
        and cluster_count != effective_expected_clusters
    ):
        raise SystemExit(
            f"Expected {effective_expected_clusters} bootstrap clusters, found {cluster_count}"
        )
    point_summaries: dict[str, dict[str, float | None]] = {}
    point_profiles: dict[str, dict[Profile, float]] = {}
    for run_name in RUN_ORDER:
        point_summaries[run_name], point_profiles[run_name] = summarize_selection(
            grouped[run_name], condition_groups
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    usable_stats = [
        statistic
        for statistic in SUMMARY_STATS
        if all(point_summaries[run][statistic] is not None for run in RUN_ORDER)
    ]
    summary_samples: dict[tuple[str, str], list[float]] = {
        (run, statistic): [] for run in RUN_ORDER for statistic in usable_stats
    }
    contrast_samples: dict[tuple[str, str], list[float]] = {
        (contrast, statistic): []
        for contrast in CONTRASTS
        for statistic in usable_stats
    }
    profile_samples: dict[tuple[str, Profile], list[float]] = {
        (run, profile): []
        for run in RUN_ORDER
        for profile in point_profiles[run]
    }
    rng = random.Random(args.seed)
    indices_path = args.output_dir / "paired_bootstrap_indices.jsonl.gz"
    bootstrap_started = time.monotonic()
    with gzip.open(indices_path, "wt", encoding="utf-8") as index_handle:
        for replicate in range(args.bootstrap_replicates):
            selected: dict[Any, list[str]] = {}
            serialized: dict[str, list[int]] = {}
            index_field = "condition_indices"
            if cluster_tables is None:
                for condition in sorted(condition_groups):
                    candidates = condition_groups[condition]
                    indices = [rng.randrange(len(candidates)) for _ in candidates]
                    selected[condition] = [candidates[index] for index in indices]
                    serialized["|".join(map(str, condition))] = indices
            else:
                index_field = "cluster_indices"
                for stratum, stratum_clusters in cluster_tables.items():
                    candidates = sorted(stratum_clusters)
                    indices = [rng.randrange(len(candidates)) for _ in candidates]
                    selected[("cluster", stratum)] = [
                        group_id
                        for index in indices
                        for group_id in stratum_clusters[candidates[index]]
                    ]
                    serialized[stratum] = indices
            index_handle.write(
                json.dumps(
                    {"replicate": replicate, index_field: serialized},
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            summaries: dict[str, dict[str, float | None]] = {}
            for run_name in RUN_ORDER:
                summaries[run_name], profiles = summarize_selection(
                    grouped[run_name], selected
                )
                for statistic in usable_stats:
                    summary_samples[(run_name, statistic)].append(
                        float(summaries[run_name][statistic])
                    )
                for profile, value in profiles.items():
                    profile_samples[(run_name, profile)].append(value)
            for contrast_name, weights in CONTRASTS.items():
                for statistic in usable_stats:
                    value = weighted_contrast(summaries, weights, statistic)
                    if value is None:
                        raise AssertionError("usable statistic produced a null contrast")
                    contrast_samples[(contrast_name, statistic)].append(value)
            completed = replicate + 1
            if completed % args.progress_every == 0 or completed == args.bootstrap_replicates:
                elapsed = time.monotonic() - bootstrap_started
                remaining = args.bootstrap_replicates - completed
                eta_seconds = elapsed / completed * remaining
                print(
                    f"bootstrap progress={completed}/{args.bootstrap_replicates} "
                    f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s",
                    flush=True,
                )

    intervals: dict[str, dict[str, Any]] = {}
    for run_name in RUN_ORDER:
        intervals[run_name] = {}
        for statistic in SUMMARY_STATS:
            estimate = point_summaries[run_name][statistic]
            if statistic not in usable_stats:
                intervals[run_name][statistic] = {
                    "estimate": estimate,
                    "ci95_low": None,
                    "ci95_high": None,
                }
                continue
            samples = summary_samples[(run_name, statistic)]
            intervals[run_name][statistic] = {
                "estimate": estimate,
                "ci95_low": percentile(samples, 0.025),
                "ci95_high": percentile(samples, 0.975),
            }

    contrasts: dict[str, Any] = {}
    p_values: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for contrast_name, weights in CONTRASTS.items():
        statistics: dict[str, Any] = {}
        for statistic in SUMMARY_STATS:
            estimate = weighted_contrast(point_summaries, weights, statistic)
            if statistic not in usable_stats:
                statistics[statistic] = {
                    "estimate": estimate,
                    "ci95_low": None,
                    "ci95_high": None,
                    "bootstrap_p_two_sided": None,
                    "holm_adjusted_p": None,
                }
                continue
            samples = contrast_samples[(contrast_name, statistic)]
            p_value = bootstrap_p_two_sided(samples)
            p_values[statistic].append((contrast_name, p_value))
            statistics[statistic] = {
                "estimate": estimate,
                "ci95_low": percentile(samples, 0.025),
                "ci95_high": percentile(samples, 0.975),
                "bootstrap_p_two_sided": p_value,
            }
        contrasts[contrast_name] = {"weights": weights, "statistics": statistics}
    for statistic, items in p_values.items():
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
                    **FACTORS[run_name],
                    "task": profile[0],
                    "target_tokens": profile[1],
                    "position_label": profile[2],
                    "answer_accuracy": estimate,
                    "ci95_low": percentile(samples, 0.025),
                    "ci95_high": percentile(samples, 0.975),
                }
            )

    trained_answers = [float(point_summaries[run]["answer_correct"]) for run in VARIANTS]
    discriminative_range = max(trained_answers) - min(trained_answers)
    saturation = min(trained_answers) >= 0.98 and discriminative_range < 0.02
    report = {
        "schema_version": "matched-factorial-analysis-v1",
        "rows_per_run": counts,
        "conditions": len(condition_groups),
        "groups": sum(len(groups) for groups in condition_groups.values()),
        "positions": list(POSITIONS),
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "unit": (
                f"cluster {cluster_key}, stratified by {cluster_strata_key or 'none'}"
                if cluster_tables is not None
                else "position-equivalent group, stratified by mode/task/filler/length"
            ),
            "cluster_key": cluster_key,
            "cluster_strata_key": cluster_strata_key,
            "cluster_source_data": (
                str(cluster_source_data.resolve())
                if cluster_source_data is not None
                else None
            ),
            "cluster_source_sha256": cluster_source_sha256,
            "cluster_count": (
                cluster_count
            ),
            "expected_clusters": effective_expected_clusters,
            "clusters_per_stratum": (
                {
                    stratum: len(stratum_clusters)
                    for stratum, stratum_clusters in cluster_tables.items()
                }
                if cluster_tables is not None
                else None
            ),
            "auto_nolima_cluster_mode": bool(auto_nolima and args.cluster_key is None),
            "indices": indices_path.name,
        },
        "run_summary_intervals": intervals,
        "contrasts": contrasts,
        "position_profiles": position_profiles,
        "benchmark_discrimination": {
            "trained_answer_accuracy_min": min(trained_answers),
            "trained_answer_accuracy_max": max(trained_answers),
            "trained_answer_accuracy_range": discriminative_range,
            "saturated_at_98pct_and_lt_2pp_range": saturation,
        },
        "generation_diagnostics": {
            run: {
                "finish_reason_counts": dict(
                    sorted(Counter(str(row.get("finish_reason", "unknown")) for row in rows).items())
                ),
                "mean_output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows)
                / len(rows),
            }
            for run, rows in rows_by_run.items()
        },
    }
    (args.output_dir / "factorial_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        args.output_dir / "factorial_summary.csv",
        (
            {
                "run_name": run,
                **FACTORS[run],
                **{stat: point_summaries[run][stat] for stat in SUMMARY_STATS},
            }
            for run in RUN_ORDER
        ),
        ("run_name", "pairing", "supervision", *SUMMARY_STATS),
    )
    write_csv(
        args.output_dir / "factorial_contrasts.csv",
        (
            {
                "contrast": contrast,
                "statistic": statistic,
                **values,
            }
            for contrast, payload in contrasts.items()
            for statistic, values in payload["statistics"].items()
        ),
        (
            "contrast",
            "statistic",
            "estimate",
            "ci95_low",
            "ci95_high",
            "bootstrap_p_two_sided",
            "holm_adjusted_p",
        ),
    )
    write_csv(
        args.output_dir / "position_profiles.csv",
        position_profiles,
        (
            "run_name",
            "pairing",
            "supervision",
            "task",
            "target_tokens",
            "position_label",
            "answer_accuracy",
            "ci95_low",
            "ci95_high",
        ),
    )
    print(f"Wrote matched 2x3 factorial analysis to {args.output_dir}")
    print(
        f"saturated={saturation} trained_answer_range={discriminative_range:.4%} "
        f"bootstrap={args.bootstrap_replicates}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
