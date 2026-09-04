#!/usr/bin/env python3
"""Aggregate audited per-seed analyses without treating predictions as training repeats."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}
PRIMARY_STATUSES = {"confirmatory", "corrective"}


def parse_analysis(value: str) -> tuple[str, int, str, Path]:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--analysis must be FAMILY:SEED:pilot|confirmatory|corrective:PATH"
        )
    family, raw_seed, status, path = parts
    if not family or status not in {"pilot", *PRIMARY_STATUSES}:
        raise argparse.ArgumentTypeError(f"Invalid analysis identity: {value}")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed: {raw_seed}") from exc
    return family, seed, status, Path(path)


def mean_interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty seed set")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {
            "n_seeds": 1,
            "mean": mean,
            "sd": None,
            "ci95_low": None,
            "ci95_high": None,
            "min": values[0],
            "max": values[0],
        }
    sd = statistics.stdev(values)
    critical = T_CRITICAL_975.get(len(values) - 1, 1.96)
    half_width = critical * sd / math.sqrt(len(values))
    return {
        "n_seeds": len(values),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "min": min(values),
        "max": max(values),
    }


def fixed_base_interval(values: Sequence[float]) -> dict[str, Any]:
    """Deduplicate a deterministic base copied into multiple seed analyses."""
    if not values:
        raise ValueError("Cannot summarize an empty fixed-base set")
    if any(not math.isclose(value, values[0], abs_tol=1e-12) for value in values[1:]):
        raise ValueError(
            "Fixed-base estimates differ across reused seed analyses; refusing to "
            "mislabel evaluation variation as training-seed variation"
        )
    summary = mean_interval([values[0]])
    summary["fixed_untrained_base"] = True
    summary["reused_analysis_copies"] = len(values)
    return summary


def extract(report: dict[str, Any]) -> tuple[str, dict[tuple[str, str], float]]:
    schema = report.get("schema_version")
    values: dict[tuple[str, str], float] = {}
    if schema == "matched-factorial-analysis-v1":
        for run, statistics in report["run_summary_intervals"].items():
            for statistic, payload in statistics.items():
                if payload.get("estimate") is not None:
                    values[(f"run:{run}", statistic)] = float(payload["estimate"])
        for contrast, payload in report["contrasts"].items():
            for statistic, estimate in payload["statistics"].items():
                if estimate.get("estimate") is not None:
                    values[(f"contrast:{contrast}", statistic)] = float(
                        estimate["estimate"]
                    )
        return "factorial", values
    if schema == "natural-transfer-analysis-v1":
        for slice_name, runs in report["run_intervals"].items():
            for run, payload in runs.items():
                values[(f"run:{run}", slice_name)] = float(payload["estimate"])
        for slice_name, contrasts in report["contrasts"].items():
            for contrast, payload in contrasts.items():
                values[(f"contrast:{contrast}", slice_name)] = float(
                    payload["estimate"]
                )
        return "natural_transfer", values
    raise ValueError(f"Unsupported analysis schema: {schema}")


def extract_position_profiles(
    report: dict[str, Any],
) -> dict[tuple[str, str, int, str], float]:
    """Extract the paper-facing position profile from one factorial seed analysis."""
    if report.get("schema_version") != "matched-factorial-analysis-v1":
        return {}
    values: dict[tuple[str, str, int, str], float] = {}
    for row in report.get("position_profiles", []):
        key = (
            str(row["run_name"]),
            str(row["task"]),
            int(row["target_tokens"]),
            str(row["position_label"]),
        )
        if key in values:
            raise ValueError(f"Duplicate position-profile cell: {key}")
        values[key] = float(row["answer_accuracy"])
    if not values:
        raise ValueError("Factorial analysis has no position profiles")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", type=parse_analysis, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    identities = [(family, seed) for family, seed, _, _ in args.analysis]
    if len(identities) != len(set(identities)):
        raise SystemExit("Duplicate family/seed analysis")
    records = []
    analysis_kind = None
    reference_keys = None
    reference_position_keys_by_family: dict[
        str, set[tuple[str, str, int, str]]
    ] = {}
    for family, seed, status, path in args.analysis:
        report = json.loads(path.read_text(encoding="utf-8"))
        kind, values = extract(report)
        position_values = extract_position_profiles(report) if kind == "factorial" else {}
        if analysis_kind is None:
            analysis_kind = kind
            reference_keys = set(values)
        elif kind != analysis_kind or set(values) != reference_keys:
            raise SystemExit("Per-seed analyses differ in kind or reported estimands")
        position_keys = set(position_values)
        if family not in reference_position_keys_by_family:
            reference_position_keys_by_family[family] = position_keys
        elif position_keys != reference_position_keys_by_family[family]:
            raise SystemExit(
                f"Per-seed analyses differ in position-profile cells within {family}"
            )
        records.append(
            {
                "family": family,
                "seed": seed,
                "status": status,
                "path": str(path.resolve()),
                "values": values,
                "position_values": position_values,
            }
        )
    assert analysis_kind is not None and reference_keys is not None
    families = sorted({record["family"] for record in records})
    primary_counts = {
        family: sum(
            record["family"] == family and record["status"] in PRIMARY_STATUSES
            for record in records
        )
        for family in families
    }
    if any(count < 2 for count in primary_counts.values()):
        raise SystemExit(
            "Each family requires at least two primary training seeds; "
            f"found {primary_counts}"
        )
    summaries: dict[str, dict[str, Any]] = {}
    long_rows = []
    for family in families:
        selected = [
            record
            for record in records
            if record["family"] == family and record["status"] in PRIMARY_STATUSES
        ]
        family_summary = {}
        for estimand, statistic in sorted(reference_keys):
            raw_values = [record["values"][(estimand, statistic)] for record in selected]
            fixed_base = estimand == "run:base"
            values = [raw_values[0]] if fixed_base else raw_values
            summary = (
                fixed_base_interval(raw_values) if fixed_base else mean_interval(values)
            )
            mean = summary["mean"]
            summary["seeds"] = (
                [] if fixed_base else [record["seed"] for record in selected]
            )
            summary["seed_estimates"] = values
            summary["direction_consistency"] = {
                "positive": sum(value > 0 for value in values),
                "zero": sum(value == 0 for value in values),
                "negative": sum(value < 0 for value in values),
                "same_sign_as_mean": sum(
                    (value > 0) == (mean > 0) for value in values if value != 0 and mean != 0
                ),
            }
            key = f"{estimand}|{statistic}"
            family_summary[key] = summary
            row_records = selected[:1] if fixed_base else selected
            for record, value in zip(row_records, values, strict=True):
                long_rows.append(
                    {
                        "family": family,
                        "seed": "" if fixed_base else record["seed"],
                        "status": "fixed_base" if fixed_base else record["status"],
                        "estimand": estimand,
                        "statistic": statistic,
                        "estimate": value,
                    }
                )
        summaries[family] = family_summary
    pilot_records = [
        {
            "family": record["family"],
            "seed": record["seed"],
            "path": record["path"],
        }
        for record in records
        if record["status"] == "pilot"
    ]
    interactions = {}
    if len(families) == 2:
        left, right = families
        interactions[f"{left}_minus_{right}"] = {
            key: summaries[left][key]["mean"] - summaries[right][key]["mean"]
            for key in summaries[left]
        }
    position_profiles = []
    if analysis_kind == "factorial":
        for family in families:
            selected = [
                record
                for record in records
                if record["family"] == family and record["status"] in PRIMARY_STATUSES
            ]
            for run_name, task, target_tokens, position_label in sorted(
                reference_position_keys_by_family[family]
            ):
                key = (run_name, task, target_tokens, position_label)
                raw_values = [record["position_values"][key] for record in selected]
                fixed_base = run_name == "base"
                values = [raw_values[0]] if fixed_base else raw_values
                summary = (
                    fixed_base_interval(raw_values)
                    if fixed_base
                    else mean_interval(values)
                )
                position_profiles.append(
                    {
                        "family": family,
                        "run_name": run_name,
                        "task": task,
                        "target_tokens": target_tokens,
                        "position_label": position_label,
                        **summary,
                        "seeds": (
                            []
                            if fixed_base
                            else [record["seed"] for record in selected]
                        ),
                        "seed_estimates": values,
                    }
                )
    report = {
        "schema_version": "seed-level-analysis-v1",
        "analysis_kind": analysis_kind,
        "inference_unit": "independently trained data/training seed",
        "primary_training_seed_summary": True,
        "confirmatory_only_primary_summary": all(
            record["status"] != "corrective" for record in records
        ),
        "primary_statuses_by_family": {
            family: sorted(
                {
                    record["status"]
                    for record in records
                    if record["family"] == family
                    and record["status"] in PRIMARY_STATUSES
                }
            )
            for family in families
        },
        "caution": (
            "Student-t intervals use only seed-level point estimates. Per-seed paired "
            "bootstrap intervals remain in source analyses; prediction rows are not "
            "treated as independent training replicates. Corrective status means the "
            "implementation was repaired after partial family results were visible and "
            "must not be described as blindly preregistered."
        ),
        "families": summaries,
        "pilot_analyses_excluded_from_primary": pilot_records,
        "descriptive_family_mean_interactions": interactions,
        "position_profiles": position_profiles,
        "source_analyses": [
            {
                key: record[key]
                for key in ("family", "seed", "status", "path")
            }
            for record in records
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seed_level_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "seed_level_estimates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("family", "seed", "status", "estimand", "statistic", "estimate"),
        )
        writer.writeheader()
        writer.writerows(long_rows)
    if position_profiles:
        with (args.output_dir / "seed_level_position_profiles.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fieldnames = (
                "family",
                "run_name",
                "task",
                "target_tokens",
                "position_label",
                "n_seeds",
                "mean",
                "sd",
                "ci95_low",
                "ci95_high",
                "min",
                "max",
                "fixed_untrained_base",
                "reused_analysis_copies",
                "seeds",
                "seed_estimates",
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in position_profiles:
                rendered = {field: row.get(field) for field in fieldnames}
                rendered["seeds"] = json.dumps(
                    row["seeds"], separators=(",", ":")
                )
                rendered["seed_estimates"] = json.dumps(
                    row["seed_estimates"], separators=(",", ":")
                )
                writer.writerow(rendered)
    print(f"Wrote seed-level {analysis_kind} aggregation to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
