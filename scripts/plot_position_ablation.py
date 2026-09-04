#!/usr/bin/env python3
"""Render publication figures from the paired ablation analysis report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_ORDER = (
    "base",
    "paired_evidence",
    "paired_answer",
    "independent_evidence",
    "independent_answer",
)
RUN_LABELS = {
    "base": "Base",
    "paired_evidence": "Paired + evidence",
    "paired_answer": "Paired + answer",
    "independent_evidence": "Independent + evidence",
    "independent_answer": "Independent + answer",
}
COLORS = {
    "base": "#4D4D4D",
    "paired_evidence": "#0072B2",
    "paired_answer": "#E69F00",
    "independent_evidence": "#009E73",
    "independent_answer": "#CC79A7",
}
LINESTYLES = {
    "base": "-",
    "paired_evidence": "-",
    "paired_answer": "--",
    "independent_evidence": "-.",
    "independent_answer": ":",
}
MARKERS = {
    "base": "o",
    "paired_evidence": "s",
    "paired_answer": "^",
    "independent_evidence": "D",
    "independent_answer": "v",
}
POSITIONS = ("p000", "p010", "p025", "p050", "p075", "p090", "p100")
POSITION_LABELS = ("0%", "10%", "25%", "50%", "75%", "90%", "100%")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_both(figure: Any, output_dir: Path, stem: str) -> dict[str, str]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    return {"png": png.name, "svg": svg.name, "pdf": pdf.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    if report.get("schema_version") != "position-ablation-analysis-v1":
        raise SystemExit("Unexpected analysis schema")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render ablation figures") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )

    profile_lookup = {
        (
            item["run_name"],
            item["task"],
            int(item["target_tokens"]),
            item["position_label"],
        ): item
        for item in report["position_profiles"]
    }
    tasks = ("kv", "two_hop")
    lengths = (8192, 32768)
    # Reserve a real header band for the title and the five-series legend.  A
    # legend anchored at the default ``upper center`` overlaps the suptitle
    # once ``bbox_inches="tight"`` is applied to the publication export.
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex=True, sharey=True)
    for row_index, task in enumerate(tasks):
        for column_index, length in enumerate(lengths):
            axis = axes[row_index][column_index]
            for run_name in RUN_ORDER:
                items = [
                    profile_lookup[(run_name, task, length, position)]
                    for position in POSITIONS
                ]
                values = [item["answer_accuracy"] for item in items]
                lower = [item["ci95_low"] for item in items]
                upper = [item["ci95_high"] for item in items]
                x_values = list(range(len(POSITIONS)))
                axis.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=COLORS[run_name],
                    alpha=0.08,
                    linewidth=0,
                )
                axis.plot(
                    x_values,
                    values,
                    label=RUN_LABELS[run_name],
                    color=COLORS[run_name],
                    linestyle=LINESTYLES[run_name],
                    marker=MARKERS[run_name],
                    markersize=4,
                    linewidth=1.7,
                )
            task_label = "Key-value retrieval" if task == "kv" else "Two-hop reasoning"
            axis.set_title(f"{task_label} · {length // 1024}K")
            axis.set_ylim(-0.02, 1.02)
            axis.set_xticks(range(len(POSITIONS)), POSITION_LABELS)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
            if row_index == 1:
                axis.set_xlabel("Evidence position in context")
            if column_index == 0:
                axis.set_ylabel("Answer accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Position robustness across the 2×2 training ablation", y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        columnspacing=1.35,
        handletextpad=0.5,
        frameon=False,
    )
    fig.text(
        0.5,
        0.005,
        "Lines average over filler types; shaded bands are paired group-bootstrap 95% CIs.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.865))
    position_files = save_both(fig, args.output_dir, "position_curves")
    plt.close(fig)

    intervals = report["run_summary_intervals"]
    summary_specs = (
        ("answer_correct", "Mean answer accuracy", True),
        ("mean_worst_position_accuracy", "Mean worst-position accuracy", True),
        ("mean_position_gap", "Mean position gap (lower is better)", False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), sharey=True)
    y_values = list(range(len(RUN_ORDER)))
    for axis, (statistic, title, accuracy_axis) in zip(axes, summary_specs, strict=True):
        for y_value, run_name in zip(y_values, RUN_ORDER, strict=True):
            item = intervals[run_name][statistic]
            estimate = item["estimate"]
            axis.errorbar(
                estimate,
                y_value,
                xerr=[
                    [estimate - item["ci95_low"]],
                    [item["ci95_high"] - estimate],
                ],
                fmt=MARKERS[run_name],
                color=COLORS[run_name],
                markersize=6,
                capsize=3,
                linewidth=1.4,
            )
        axis.set_title(title)
        axis.set_xlim(-0.02, 1.02)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_xlabel("Accuracy" if accuracy_axis else "Max − min accuracy")
    axes[0].set_yticks(y_values, [RUN_LABELS[name] for name in RUN_ORDER])
    axes[0].invert_yaxis()
    fig.suptitle("Training effects must improve the weakest position without hiding the gap")
    fig.text(
        0.5,
        0.01,
        "Points are full-pilot estimates; bars are paired, condition-stratified bootstrap 95% CIs.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    summary_files = save_both(fig, args.output_dir, "ablation_summary")
    plt.close(fig)

    metadata = {
        "schema_version": "position-ablation-figures-v1",
        "analysis": str(args.analysis.resolve()),
        "analysis_sha256": sha256_file(args.analysis),
        "matplotlib_version": matplotlib.__version__,
        "files": {"position_curves": position_files, "ablation_summary": summary_files},
        "accessibility": {
            "encoding": "Series use color, line style, and marker shape redundantly.",
            "position_curves_alt": (
                "Four panels compare answer accuracy across seven evidence positions for "
                "key-value and two-hop tasks at 8K and 32K. Five model variants have "
                "paired-bootstrap confidence bands."
            ),
            "ablation_summary_alt": (
                "Three dot-and-interval panels compare mean answer accuracy, average "
                "worst-position accuracy, and average position gap for the base model "
                "and four training ablations."
            ),
        },
    }
    (args.output_dir / "figures.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote publication figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
