#!/usr/bin/env python3
"""Render publication-ready training diagnostics from exported Trainer metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RUN_STYLES = {
    "independent_answer": ("#0072B2", "--", "o"),
    "paired_answer": ("#0072B2", "-", "o"),
    "independent_evidence_id": ("#E69F00", "--", "s"),
    "paired_evidence_id": ("#E69F00", "-", "s"),
    "independent_evidence": ("#009E73", "--", "^"),
    "paired_evidence": ("#009E73", "-", "^"),
}


def read_metrics(path: Path) -> dict[str, list[dict[str, float]]]:
    by_variant: dict[str, list[dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            variant = raw["variant"]
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key == "variant" or value in (None, ""):
                    continue
                row[key] = float(value)
            by_variant.setdefault(variant, []).append(row)
    for rows in by_variant.values():
        rows.sort(key=lambda row: row["step"])
    if not by_variant:
        raise ValueError(f"No metric rows in {path}")
    return by_variant


def ewma(values: list[float], span: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def series(rows: list[dict[str, float]], field: str) -> tuple[list[float], list[float]]:
    selected = [(row["step"], row[field]) for row in rows if field in row]
    return [item[0] for item in selected], [item[1] for item in selected]


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--ema-span", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="QLoRA ablation training diagnostics")
    parser.add_argument("--scheduler-horizon", type=int)
    parser.add_argument("--warmup-steps", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.ema_span <= 0 or args.dpi <= 0:
        raise SystemExit("--ema-span and --dpi must be positive")
    if (args.scheduler_horizon is None) != (args.warmup_steps is None):
        raise SystemExit("--scheduler-horizon and --warmup-steps must be provided together")
    if args.scheduler_horizon is not None and not (
        0 <= args.warmup_steps < args.scheduler_horizon
    ):
        raise SystemExit("warmup steps must be non-negative and below the scheduler horizon")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render training curves") from exc

    metrics_path = args.diagnostics_dir / "training_metrics.csv"
    by_variant = read_metrics(metrics_path)
    figures_dir = args.diagnostics_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": args.dpi,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    scheduler_title = "Cosine learning-rate schedule"
    if args.scheduler_horizon is not None:
        scheduler_title = (
            f"Learning rate ({args.scheduler_horizon:,}-step cosine horizon; "
            f"{args.warmup_steps}-step warmup)"
        )
    panels = (
        (axes[0, 0], "loss", "Completion-only training loss", "Loss", True, True),
        (axes[0, 1], "learning_rate", scheduler_title, "Learning rate", False, False),
        (axes[1, 0], "grad_norm", "Gradient norm", "Gradient norm", True, True),
        (axes[1, 1], "mean_token_accuracy", "Completion-token accuracy", "Accuracy", False, True),
    )
    for axis, field, title, ylabel, log_scale, smooth in panels:
        for variant, rows in by_variant.items():
            steps, values = series(rows, field)
            if not values:
                continue
            color, linestyle, marker = RUN_STYLES.get(
                variant, (None, "-", None)
            )
            label = variant.replace("_", " ")
            if smooth:
                axis.plot(
                    steps,
                    values,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.17,
                    linewidth=0.45,
                )
                axis.plot(
                    steps,
                    ewma(values, args.ema_span),
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=max(1, len(steps) // 5),
                    markersize=2.8,
                    linewidth=1.45,
                    label=f"{label} (EMA {args.ema_span})",
                )
            else:
                axis.plot(
                    steps,
                    values,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=max(1, len(steps) // 5),
                    markersize=2.8,
                    linewidth=1.35,
                    label=label,
                )
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="major", color="#D0D0D0", linewidth=0.5, alpha=0.8)
        axis.grid(True, which="minor", color="#E8E8E8", linewidth=0.35, alpha=0.55)
        if log_scale:
            axis.set_yscale("log")
        if field == "learning_rate":
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if field == "mean_token_accuracy":
            axis.set_ylim(0.0, 1.01)
        axis.legend(loc="best", frameon=False)
    fig.suptitle(args.title, fontsize=12)
    png_path = figures_dir / "training_curves.png"
    svg_path = figures_dir / "training_curves.svg"
    pdf_path = figures_dir / "training_curves.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    alt_path = figures_dir / "training_curves.alt.txt"
    schedule_description = (
        f"a {args.scheduler_horizon:,}-step cosine horizon with "
        f"{args.warmup_steps} warmup steps"
        if args.scheduler_horizon is not None
        else "the recorded learning-rate schedule"
    )
    alt_path.write_text(
        (
            f"Four-panel training diagnostic for {args.title}. Panels show completion-only "
            "loss and gradient norm on logarithmic scales, learning rate under "
            f"{schedule_description}, and completion-token accuracy by optimizer step. "
            "Color encodes supervision: blue answer, orange evidence ID, and green exact "
            "evidence. Dashed lines denote independent position sampling and solid lines "
            "denote paired cross-position construction; marker shape redundantly encodes "
            "supervision. Faint lines are raw step values and stronger lines are "
            f"EMA-{args.ema_span} trends. Exact values are in training_metrics.csv.\n"
        ),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "training-curve-figure-v2",
        "source": str(metrics_path.resolve()),
        "source_sha256": sha256_file(metrics_path),
        "variants": sorted(by_variant),
        "rows_by_variant": {key: len(value) for key, value in sorted(by_variant.items())},
        "ema_span": args.ema_span,
        "title": args.title,
        "scheduler_horizon_steps": args.scheduler_horizon,
        "warmup_steps": args.warmup_steps,
        "loss_scale": "log",
        "gradient_norm_scale": "log",
        "png_dpi": args.dpi,
        "matplotlib_version": matplotlib.__version__,
        "outputs": {
            path.suffix.lstrip("."): {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in (png_path, svg_path, pdf_path, alt_path)
        },
        "visual_encoding": {
            "color": "supervision (answer, evidence ID, exact evidence)",
            "line_style": "position construction (independent dashed, paired solid)",
            "marker": "supervision, redundant with color",
        },
        "note": "Faint lines are raw per-step values; emphasized lines are EMA-smoothed.",
    }
    write_json_atomic(figures_dir / "training_curves.figure.json", metadata)
    print(f"Wrote {png_path}, {svg_path}, {pdf_path}, and {alt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
