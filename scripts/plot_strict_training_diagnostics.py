#!/usr/bin/env python3
"""Render a cross-family, cross-seed strict checkpoint-96 training figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


EXPECTED_FAMILIES = ("Qwen2.5-7B", "Mistral-7B-v0.3")
EXPECTED_SEEDS = (20260825, 20260826, 20260827)
RUN_STYLES = {
    "independent_answer": ("#0072B2", "--"),
    "paired_answer": ("#0072B2", "-"),
    "independent_evidence_id": ("#E69F00", "--"),
    "paired_evidence_id": ("#E69F00", "-"),
    "independent_evidence": ("#009E73", "--"),
    "paired_evidence": ("#009E73", "-"),
}
METRICS = ("loss", "learning_rate", "grad_norm", "mean_token_accuracy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_source(value: str) -> tuple[str, int, Path]:
    try:
        family, seed_text, path_text = value.split(":", 2)
        seed = int(seed_text)
    except (ValueError, TypeError) as exc:
        raise ValueError("--metrics must be FAMILY:SEED:CSV") from exc
    if not family or not path_text:
        raise ValueError("--metrics must be FAMILY:SEED:CSV")
    return family, seed, Path(path_text).resolve()


def load_metrics(
    specifications: list[str], expected_steps: int
) -> tuple[dict[tuple[str, int, str, int], dict[str, float]], list[dict[str, Any]]]:
    rows: dict[tuple[str, int, str, int], dict[str, float]] = {}
    sources: list[dict[str, Any]] = []
    source_keys: set[tuple[str, int]] = set()
    for specification in specifications:
        family, seed, path = parse_source(specification)
        if family not in EXPECTED_FAMILIES or seed not in EXPECTED_SEEDS:
            raise ValueError(f"Unexpected strict family/seed: {family}:{seed}")
        if (family, seed) in source_keys:
            raise ValueError(f"Duplicate metrics source: {family}:{seed}")
        if not path.is_file():
            raise ValueError(f"Metrics CSV is missing: {path}")
        source_keys.add((family, seed))
        source_rows = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"variant", "step", *METRICS}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Metrics CSV lacks required columns: {path}")
            for raw in reader:
                variant = raw["variant"]
                if variant not in RUN_STYLES:
                    raise ValueError(f"Unexpected training variant: {variant}")
                step = int(raw["step"])
                values = {metric: float(raw[metric]) for metric in METRICS}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError(f"Non-finite metric at {family}/{seed}/{variant}/{step}")
                if values["loss"] <= 0 or values["grad_norm"] <= 0:
                    raise ValueError(f"Log-scale metric is not positive at step {step}")
                key = family, seed, variant, step
                if key in rows:
                    raise ValueError(f"Duplicate training metric row: {key}")
                rows[key] = values
                source_rows += 1
        expected_rows = len(RUN_STYLES) * expected_steps
        if source_rows != expected_rows:
            raise ValueError(
                f"{family}/{seed} has {source_rows} metric rows; expected {expected_rows}"
            )
        for variant in RUN_STYLES:
            steps = {
                step
                for row_family, row_seed, row_variant, step in rows
                if (row_family, row_seed, row_variant) == (family, seed, variant)
            }
            if steps != set(range(1, expected_steps + 1)):
                raise ValueError(f"Incomplete step trace: {family}/{seed}/{variant}")
        sources.append(
            {
                "family": family,
                "seed": seed,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": source_rows,
            }
        )
    expected_keys = {
        (family, seed) for family in EXPECTED_FAMILIES for seed in EXPECTED_SEEDS
    }
    if source_keys != expected_keys:
        missing = sorted(expected_keys - source_keys)
        extra = sorted(source_keys - expected_keys)
        raise ValueError(f"Strict metric source matrix differs: missing={missing} extra={extra}")
    return rows, sorted(sources, key=lambda item: (item["family"], item["seed"]))


def aggregate_rows(
    rows: dict[tuple[str, int, str, int], dict[str, float]], expected_steps: int
) -> list[dict[str, Any]]:
    aggregated = []
    for family in EXPECTED_FAMILIES:
        for variant in RUN_STYLES:
            for step in range(1, expected_steps + 1):
                for metric in METRICS:
                    values = [
                        rows[(family, seed, variant, step)][metric]
                        for seed in EXPECTED_SEEDS
                    ]
                    aggregated.append(
                        {
                            "family": family,
                            "variant": variant,
                            "step": step,
                            "metric": metric,
                            "mean": statistics.fmean(values),
                            "min": min(values),
                            "max": max(values),
                            "n_seeds": len(values),
                        }
                    )
    return aggregated


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("family", "variant", "step", "metric", "mean", "min", "max", "n_seeds"),
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def select_series(
    aggregated: list[dict[str, Any]], family: str, variant: str, metric: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in aggregated
        if row["family"] == family
        and row["variant"] == variant
        and row["metric"] == metric
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default="strict_training_diagnostics")
    parser.add_argument("--expected-steps", type=int, default=96)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--scheduler-horizon", type=int, default=2000)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if not (
        args.expected_steps > 0
        and 0 < args.warmup_steps < args.expected_steps
        and args.scheduler_horizon > args.expected_steps
        and args.dpi > 0
    ):
        raise SystemExit("Invalid strict step, warmup, horizon, or DPI setting")
    try:
        rows, sources = load_metrics(args.metrics, args.expected_steps)
        aggregated = aggregate_rows(rows, args.expected_steps)
    except (OSError, ValueError, csv.Error) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render training diagnostics") from exc

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{args.basename}.csv"
    write_table(table_path, aggregated)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": args.dpi,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(12.5, 12.0),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    metric_layout = (
        ("loss", "Completion-only loss", True),
        ("learning_rate", "Learning rate", False),
        ("grad_norm", "Gradient norm", True),
        ("mean_token_accuracy", "Completion-token accuracy", False),
    )
    legend_handles = []
    for column, family in enumerate(EXPECTED_FAMILIES):
        axes[0, column].set_title(family)
        for row_index, (metric, ylabel, logarithmic) in enumerate(metric_layout):
            axis = axes[row_index, column]
            if metric == "learning_rate":
                schedule = select_series(
                    aggregated, family, "independent_answer", metric
                )
                for variant in RUN_STYLES:
                    comparison = select_series(aggregated, family, variant, metric)
                    if any(
                        abs(float(left["mean"]) - float(right["mean"])) > 1e-12
                        for left, right in zip(schedule, comparison, strict=True)
                    ):
                        raise SystemExit(f"Learning-rate schedule differs in {family}")
                axis.plot(
                    [item["step"] for item in schedule],
                    [item["mean"] for item in schedule],
                    color="#333333",
                    linewidth=1.6,
                    label="Frozen schedule",
                )
                axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            else:
                for variant, (color, linestyle) in RUN_STYLES.items():
                    series = select_series(aggregated, family, variant, metric)
                    steps = [int(item["step"]) for item in series]
                    means = [float(item["mean"]) for item in series]
                    lows = [float(item["min"]) for item in series]
                    highs = [float(item["max"]) for item in series]
                    axis.fill_between(steps, lows, highs, color=color, alpha=0.08)
                    axis.plot(
                        steps,
                        means,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.35,
                    )
            axis.axvline(
                args.warmup_steps,
                color="#666666",
                linestyle=":",
                linewidth=0.9,
                alpha=0.85,
            )
            axis.set_ylabel(ylabel)
            axis.grid(True, which="major", color="#D5D5D5", linewidth=0.45, alpha=0.75)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if logarithmic:
                axis.set_yscale("log")
            if metric == "mean_token_accuracy":
                axis.set_ylim(0.0, 1.01)
        axes[-1, column].set_xlabel("Optimizer step")

    for variant, (color, linestyle) in RUN_STYLES.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                label=variant.replace("_", " "),
            )
        )
    legend_handles.extend(
        (
            Line2D([0], [0], color="#333333", linewidth=1.6, label="frozen LR schedule"),
            Line2D([0], [0], color="#666666", linestyle=":", label="warmup boundary"),
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Strict 96-step QLoRA diagnostics across three independent seeds",
        fontsize=13,
    )
    output_paths = {
        suffix: output_dir / f"{args.basename}.{suffix}"
        for suffix in ("pdf", "svg", "png")
    }
    fig.savefig(output_paths["pdf"], bbox_inches="tight")
    fig.savefig(output_paths["svg"], bbox_inches="tight")
    fig.savefig(output_paths["png"], dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    descriptions = []
    for family in EXPECTED_FAMILIES:
        final_losses = [
            float(select_series(aggregated, family, variant, "loss")[-1]["mean"])
            for variant in RUN_STYLES
        ]
        final_accuracies = [
            float(
                select_series(
                    aggregated, family, variant, "mean_token_accuracy"
                )[-1]["mean"]
            )
            for variant in RUN_STYLES
        ]
        descriptions.append(
            f"{family} mean final loss ranges from {min(final_losses):.3g} to "
            f"{max(final_losses):.3g}; mean final token accuracy ranges from "
            f"{min(final_accuracies):.4f} to {max(final_accuracies):.4f}."
        )
    alt_path = output_dir / f"{args.basename}.alt.txt"
    write_text_atomic(
        alt_path,
        (
            "Eight-panel strict-training diagnostic. Columns compare Qwen2.5-7B and "
            "Mistral-7B-v0.3; rows show completion-only loss, the frozen learning-rate "
            "schedule, gradient norm, and completion-token accuracy over optimizer steps "
            f"1--{args.expected_steps}. Each treatment line is the mean over three seeds; "
            "the translucent envelope is the seed minimum-to-maximum range. Color encodes "
            "supervision and solid versus dashed lines encode paired versus independent "
            f"construction. The dotted line marks warmup step {args.warmup_steps}. "
            + " ".join(descriptions)
            + f" Exact plotted values are in {table_path.name}.\n"
        ),
    )
    all_outputs = {**output_paths, "csv": table_path, "alt": alt_path}
    manifest_path = output_dir / f"{args.basename}.manifest.json"
    manifest = {
        "schema_version": "strict-training-diagnostics-figure-v1",
        "status": "validated",
        "families": list(EXPECTED_FAMILIES),
        "seeds": list(EXPECTED_SEEDS),
        "variants": list(RUN_STYLES),
        "expected_steps": args.expected_steps,
        "warmup_steps": args.warmup_steps,
        "scheduler_horizon": args.scheduler_horizon,
        "aggregation": "arithmetic mean with seed min-max envelope; descriptive only",
        "sources": sources,
        "outputs": {
            label: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(all_outputs.items())
        },
        "table_rows": len(aggregated),
        "matplotlib_version": matplotlib.__version__,
    }
    write_text_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"Rendered strict training diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
