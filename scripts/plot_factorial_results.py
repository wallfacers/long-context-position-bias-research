#!/usr/bin/env python3
"""Render publication figures for the matched 2x3 factorial analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POSITIONS = ("p000", "p010", "p025", "p050", "p075", "p090", "p100")
POSITION_LABELS = ("0%", "10%", "25%", "50%", "75%", "90%", "100%")
SUPERVISIONS = ("answer", "evidence_id", "evidence")
SUPERVISION_LABELS = {
    "answer": "Answer only",
    "evidence_id": "Answer + evidence ID",
    "evidence": "Answer + exact evidence",
}
PAIRING_COLORS = {
    "base": "#4D4D4D",
    "independent": "#0072B2",
    "paired": "#D55E00",
}
PAIRING_LABELS = {
    "base": "Base",
    "independent": "Independent positions",
    "paired": "Paired positions",
}
LINESTYLES = {"base": ":", "independent": "--", "paired": "-"}
MARKERS = {"base": "o", "independent": "^", "paired": "s"}
SUMMARY_RUN_ORDER = (
    "base",
    "independent_answer",
    "paired_answer",
    "independent_evidence_id",
    "paired_evidence_id",
    "independent_evidence",
    "paired_evidence",
)
RUN_LABELS = {
    "base": "Base",
    "independent_answer": "Independent + answer",
    "paired_answer": "Paired + answer",
    "independent_evidence_id": "Independent + evidence ID",
    "paired_evidence_id": "Paired + evidence ID",
    "independent_evidence": "Independent + exact evidence",
    "paired_evidence": "Paired + exact evidence",
}
SUPERVISION_COLORS = {
    None: "#4D4D4D",
    "answer": "#E69F00",
    "evidence_id": "#009E73",
    "evidence": "#0072B2",
}


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


def run_for(pairing: str, supervision: str) -> str:
    return f"{pairing}_{supervision}"


def profile_label(task: str, target_tokens: int) -> str:
    if target_tokens >= 1024 and target_tokens % 1024 == 0:
        length = f"{target_tokens // 1024}K"
    else:
        length = f"{target_tokens:,} tokens"
    return f"{task.replace('_', ' ')} · {length}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    if report.get("schema_version") != "matched-factorial-analysis-v1":
        raise SystemExit("Unexpected factorial analysis schema")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render factorial figures") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )

    profiles = report["position_profiles"]
    lookup = {
        (
            item["run_name"],
            item["task"],
            int(item["target_tokens"]),
            item["position_label"],
        ): item
        for item in profiles
    }
    profile_keys = sorted(
        {(item["task"], int(item["target_tokens"])) for item in profiles},
        key=lambda item: (item[1], item[0]),
    )
    if not profile_keys:
        raise SystemExit("No position profiles in analysis")
    fig_height = max(3.4, 2.25 * len(profile_keys) + 1.4)
    fig, axes = plt.subplots(
        len(profile_keys),
        len(SUPERVISIONS),
        figsize=(12.2, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_values = list(range(len(POSITIONS)))
    for row_index, (task, target_tokens) in enumerate(profile_keys):
        for column_index, supervision in enumerate(SUPERVISIONS):
            axis = axes[row_index][column_index]
            for pairing in ("base", "independent", "paired"):
                run_name = "base" if pairing == "base" else run_for(pairing, supervision)
                items = [lookup[(run_name, task, target_tokens, pos)] for pos in POSITIONS]
                values = [item["answer_accuracy"] for item in items]
                lower = [item["ci95_low"] for item in items]
                upper = [item["ci95_high"] for item in items]
                axis.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=PAIRING_COLORS[pairing],
                    alpha=0.08,
                    linewidth=0,
                )
                axis.plot(
                    x_values,
                    values,
                    color=PAIRING_COLORS[pairing],
                    linestyle=LINESTYLES[pairing],
                    marker=MARKERS[pairing],
                    markersize=3.7,
                    linewidth=1.6,
                    label=PAIRING_LABELS[pairing],
                )
            if row_index == 0:
                axis.set_title(SUPERVISION_LABELS[supervision])
            axis.set_ylim(-0.02, 1.02)
            axis.set_xticks(x_values, POSITION_LABELS)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
            if column_index == 0:
                axis.set_ylabel(f"{profile_label(task, target_tokens)}\nAnswer accuracy")
            if row_index == len(profile_keys) - 1:
                axis.set_xlabel("Evidence position")
    handles = [
        Line2D(
            [0],
            [0],
            color=PAIRING_COLORS[pairing],
            linestyle=LINESTYLES[pairing],
            marker=MARKERS[pairing],
            linewidth=1.7,
            markersize=4,
            label=PAIRING_LABELS[pairing],
        )
        for pairing in ("base", "independent", "paired")
    ]
    fig.suptitle("Position robustness separates pairing from supervision granularity", y=0.992)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.006,
        "Bands are paired, condition-stratified bootstrap 95% CIs; panels share the same 0–100% scale.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    position_files = save_both(fig, args.output_dir, "factorial_position_curves")
    plt.close(fig)

    intervals = report["run_summary_intervals"]
    summary_specs = (
        ("answer_correct", "Mean answer accuracy", "Accuracy"),
        ("mean_worst_answer_accuracy", "Worst-position accuracy", "Accuracy"),
        ("mean_answer_position_gap", "Position gap", "Max − min accuracy"),
        ("evidence_quotes_correct", "Exact-quote accuracy", "Accuracy"),
    )
    fig, axes = plt.subplots(1, len(summary_specs), figsize=(14.2, 5.2), sharey=True)
    y_values = list(range(len(SUMMARY_RUN_ORDER)))
    for axis, (statistic, title, xlabel) in zip(axes, summary_specs, strict=True):
        for y_value, run_name in zip(y_values, SUMMARY_RUN_ORDER, strict=True):
            item = intervals[run_name][statistic]
            estimate = item["estimate"]
            if estimate is None:
                axis.text(0.5, y_value, "N/A", ha="center", va="center", color="#666666")
                continue
            pairing = report.get("factors", {}).get(run_name, {}).get("pairing")
            if run_name == "base":
                pairing = "base"
                supervision = None
            else:
                pairing, supervision = run_name.split("_", 1)
            axis.errorbar(
                estimate,
                y_value,
                xerr=[[estimate - item["ci95_low"]], [item["ci95_high"] - estimate]],
                fmt=MARKERS[pairing],
                color=SUPERVISION_COLORS[supervision],
                fillstyle="full" if pairing in ("base", "paired") else "none",
                markersize=6,
                capsize=3,
                linewidth=1.35,
            )
        axis.set_title(title)
        axis.set_xlim(-0.02, 1.02)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.7)
        axis.set_xlabel(xlabel)
    axes[0].set_yticks(y_values, [RUN_LABELS[name] for name in SUMMARY_RUN_ORDER])
    axes[0].invert_yaxis()
    fig.suptitle("Matched training effects on correctness, robustness, and verifiability")
    fig.text(
        0.5,
        0.008,
        "Color encodes supervision; filled vs. hollow markers encode paired vs. independent position construction.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    summary_files = save_both(fig, args.output_dir, "factorial_summary")
    plt.close(fig)

    metadata = {
        "schema_version": "matched-factorial-figures-v1",
        "analysis": str(args.analysis.resolve()),
        "analysis_sha256": sha256_file(args.analysis),
        "matplotlib_version": matplotlib.__version__,
        "files": {"position_curves": position_files, "summary": summary_files},
        "accessibility": {
            "encoding": (
                "Position curves use color, line style, and marker shape redundantly. "
                "Summary points use supervision colors and paired/independent marker fill."
            ),
            "position_curves_alt": (
                "A matrix of plots compares base, independent-position, and paired-position "
                "training across seven evidence locations for answer-only, evidence-ID, and "
                "exact-evidence supervision. All panels share a zero-to-one accuracy scale."
            ),
            "summary_alt": (
                "Four dot-and-interval panels compare seven conditions on mean answer "
                "accuracy, worst-position accuracy, position gap, and exact-quote accuracy."
            ),
            "table_alternatives": ["factorial_summary.csv", "position_profiles.csv"],
        },
    }
    (args.output_dir / "figures.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote matched factorial publication figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
