#!/usr/bin/env python3
"""Render strict-primary seed-level position curves across model families."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


POSITIONS = ("p000", "p010", "p025", "p050", "p075", "p090", "p100")
POSITION_TICKS = ("0", "10", "25", "50", "75", "90", "100")
RUN_STYLES = {
    "base": ("Base", "#333333", ":", "x"),
    "independent_answer": ("Independent + answer", "#0072B2", "--", "o"),
    "paired_answer": ("Paired + answer", "#0072B2", "-", "o"),
    "independent_evidence_id": ("Independent + evidence ID", "#E69F00", "--", "s"),
    "paired_evidence_id": ("Paired + evidence ID", "#E69F00", "-", "s"),
    "independent_evidence": ("Independent + exact evidence", "#009E73", "--", "^"),
    "paired_evidence": ("Paired + exact evidence", "#009E73", "-", "^"),
}
EXPECTED_PRIMARY_STATUSES = {
    "Qwen2.5-7B": ["corrective"],
    "Mistral-7B-v0.3": ["confirmatory"],
}
NOLIMA_CASE_WEIGHTS = {
    "nolima_onehop": 2,
    "nolima_twohop": 6,
    "nolima_twohop2": 2,
}
T_CRITICAL_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
LENGTH_LABELS = {1024: "1K", 8192: "8K", 32000: "32K"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["family"]), str(row["task"]), int(row["target_tokens"])


def mean_interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot plot an empty seed set")
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


def aggregate_tasks(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create the readable main figure while preserving task-level source rows.

    NoLiMa-Hard contains 2/6/2 semantic cases in its three task strata.  Books,
    lengths, and positions are balanced inside every case, so weighting task-level
    accuracies by semantic-case count exactly recovers the case-weighted accuracy.
    A one-task input remains useful for unit tests and downstream custom subsets.
    """
    tasks = sorted({str(row["task"]) for row in rows})
    if len(tasks) == 1:
        weights = {tasks[0]: 1}
        output_task = tasks[0]
    elif set(tasks) == set(NOLIMA_CASE_WEIGHTS):
        weights = {task: NOLIMA_CASE_WEIGHTS[task] for task in tasks}
        output_task = "case_weighted_nolima_hard"
    else:
        raise ValueError(
            "Cannot infer task weights for the paper figure: " + ", ".join(tasks)
        )

    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["family"]),
                str(row["run_name"]),
                int(row["target_tokens"]),
                str(row["position_label"]),
            )
        ].append(row)

    aggregated = []
    for (family, run_name, target_tokens, position_label), task_rows in sorted(
        grouped.items()
    ):
        by_task = {str(row["task"]): row for row in task_rows}
        if set(by_task) != set(weights):
            raise ValueError(
                "Incomplete task grid for "
                f"{family}/{run_name}/{target_tokens}/{position_label}"
            )
        fixed_base = run_name == "base"
        if fixed_base:
            seed_ids: list[int | None] = [None]
        else:
            seed_lists = [list(by_task[task].get("seeds", [])) for task in tasks]
            if not seed_lists[0] or any(item != seed_lists[0] for item in seed_lists[1:]):
                raise ValueError(
                    "Task rows have inconsistent confirmatory seeds for "
                    f"{family}/{run_name}/{target_tokens}/{position_label}"
                )
            seed_ids = [int(seed) for seed in seed_lists[0]]

        seed_values = []
        for seed_index, _seed in enumerate(seed_ids):
            numerator = 0.0
            denominator = 0
            for task in tasks:
                estimates = list(by_task[task].get("seed_estimates", []))
                expected = 1 if fixed_base else len(seed_ids)
                if len(estimates) != expected:
                    raise ValueError(
                        "Task row has an invalid seed-estimate count for "
                        f"{family}/{run_name}/{task}/{target_tokens}/{position_label}"
                    )
                numerator += weights[task] * float(estimates[seed_index])
                denominator += weights[task]
            seed_values.append(numerator / denominator)
        summary = mean_interval(seed_values)
        aggregated.append(
            {
                "family": family,
                "run_name": run_name,
                "task": output_task,
                "target_tokens": target_tokens,
                "position_label": position_label,
                **summary,
                "fixed_untrained_base": fixed_base,
                "seeds": [] if fixed_base else seed_ids,
                "seed_estimates": seed_values,
            }
        )
    return aggregated, weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default="factorial_position_curves")
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != "seed-level-analysis-v1"
        or report.get("analysis_kind") != "factorial"
        or report.get("primary_training_seed_summary") is not True
        or report.get("confirmatory_only_primary_summary") is not False
        or report.get("primary_statuses_by_family") != EXPECTED_PRIMARY_STATUSES
    ):
        raise SystemExit(
            "Expected strict Qwen-corrective plus Mistral-confirmatory factorial analysis"
        )
    source_rows = report.get("position_profiles", [])
    if not source_rows:
        raise SystemExit("Seed-level analysis has no position profiles")
    source_tasks = sorted({str(row["task"]) for row in source_rows})
    try:
        rows, task_weights = aggregate_tasks(source_rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unexpected = sorted({str(row["run_name"]) for row in rows} - set(RUN_STYLES))
    if unexpected:
        raise SystemExit(f"Unrecognized run names: {unexpected}")
    expected_positions = set(POSITIONS)
    panels = sorted({panel_key(row) for row in rows})
    for panel in panels:
        panel_rows = [row for row in rows if panel_key(row) == panel]
        run_positions: dict[str, set[str]] = {}
        for row in panel_rows:
            run_positions.setdefault(str(row["run_name"]), set()).add(
                str(row["position_label"])
            )
        if set(run_positions) != set(RUN_STYLES) or any(
            positions != expected_positions for positions in run_positions.values()
        ):
            raise SystemExit(f"Incomplete run/position grid for panel {panel}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render seed-level figures") from exc

    row_keys = sorted({(family, task) for family, task, _ in panels})
    lengths = sorted({length for _, _, length in panels})
    figure, axes = plt.subplots(
        len(row_keys),
        len(lengths),
        figsize=(4.6 * len(lengths), 3.25 * len(row_keys)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    by_cell = {
        (
            str(row["family"]),
            str(row["task"]),
            int(row["target_tokens"]),
            str(row["run_name"]),
            str(row["position_label"]),
        ): row
        for row in rows
    }
    position_index = {position: index for index, position in enumerate(POSITIONS)}
    handles = []
    labels = []
    for row_index, (family, task) in enumerate(row_keys):
        for column_index, length in enumerate(lengths):
            axis = axes[row_index][column_index]
            if (family, task, length) not in panels:
                axis.set_visible(False)
                continue
            for run_name, (label, color, linestyle, marker) in RUN_STYLES.items():
                series = [
                    by_cell[(family, task, length, run_name, position)]
                    for position in POSITIONS
                ]
                means = [float(item["mean"]) for item in series]
                lower = [
                    max(0.0, float(item["ci95_low"]))
                    if item.get("ci95_low") is not None
                    else value
                    for item, value in zip(series, means, strict=True)
                ]
                upper = [
                    min(1.0, float(item["ci95_high"]))
                    if item.get("ci95_high") is not None
                    else value
                    for item, value in zip(series, means, strict=True)
                ]
                x_values = list(range(len(POSITIONS)))
                line = axis.plot(
                    x_values,
                    means,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4,
                    linewidth=1.7,
                    label=label,
                )[0]
                axis.fill_between(x_values, lower, upper, color=color, alpha=0.07)
                if row_index == 0 and column_index == 0:
                    handles.append(line)
                    labels.append(label)
            length_label = LENGTH_LABELS.get(length, f"{length:,}")
            axis.set_title(f"{length_label} tokens", fontsize=11)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks(range(len(POSITIONS)), POSITION_TICKS)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if column_index == 0:
                task_label = (
                    "Case-weighted NoLiMa-Hard"
                    if task == "case_weighted_nolima_hard"
                    else task.replace("_", " ").title()
                )
                axis.set_ylabel(f"{family}\n{task_label}\nAnswer accuracy")
            if row_index == len(row_keys) - 1:
                axis.set_xlabel("Evidence position (% of context)")
    figure.suptitle(
        "Strict-primary position robustness across model families and context lengths",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("pdf", "svg", "png"):
        path = args.output_dir / f"{args.basename}.{suffix}"
        figure.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        outputs[suffix] = path
    plt.close(figure)

    table_path = args.output_dir / f"{args.basename}.csv"
    table_fields = (
        "family",
        "task",
        "target_tokens",
        "run_name",
        "position_label",
        "n_seeds",
        "mean",
        "ci95_low",
        "ci95_high",
        "seed_estimates",
    )
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["family"]),
                str(item["task"]),
                int(item["target_tokens"]),
                list(RUN_STYLES).index(str(item["run_name"])),
                position_index[str(item["position_label"])],
            ),
        ):
            writer.writerow({field: row.get(field) for field in table_fields})

    families = sorted({str(row["family"]) for row in rows})
    alt_text = (
        "Small-multiple line chart of strict-primary case-weighted mean answer accuracy "
        "across seven relative evidence positions for base and six matched training "
        "treatments. "
        f"Rows cover model families {', '.join(families)}; "
        f"columns cover token lengths {', '.join(str(value) for value in lengths)}. "
        "The three NoLiMa-Hard task strata are weighted by their 2, 6, and 2 "
        "underlying semantic cases. "
        "Color denotes supervision, solid versus dashed lines denote paired versus "
        "independent position construction, and translucent bands show clipped 95% "
        "Student-t intervals over independently trained primary seeds. Qwen is a "
        "post-hoc corrective replication and Mistral is prospective confirmatory under "
        "the corrected protocol. The fixed "
        "untrained base is deduplicated and therefore has no seed interval. Exact, "
        "unclipped interval values are available in the companion CSV."
    )
    alt_path = args.output_dir / f"{args.basename}.alt.txt"
    alt_path.write_text(alt_text + "\n", encoding="utf-8")
    manifest_path = args.output_dir / f"{args.basename}.manifest.json"
    files = {**outputs, "table": table_path, "alt_text": alt_path}
    manifest = {
        "schema_version": "seed-level-factorial-figure-v2",
        "status": "validated",
        "confirmatory_only": False,
        "corrective_plus_confirmatory_primary": True,
        "primary_statuses_by_family": EXPECTED_PRIMARY_STATUSES,
        "analysis_path": str(args.analysis.resolve()),
        "analysis_sha256": sha256_file(args.analysis),
        "families": families,
        "source_tasks": source_tasks,
        "task_case_weights": task_weights,
        "aggregation": "case-weighted across NoLiMa-Hard task strata before seed-level Student-t intervals",
        "lengths": lengths,
        "positions": list(POSITIONS),
        "interval_note": "Student-t seed intervals are clipped only in the plot; CSV is exact.",
        "files": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in files.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Rendered strict-primary seed-level position figure to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
