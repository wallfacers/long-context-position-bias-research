#!/usr/bin/env python3
"""Render publication figures for natural long-context transfer results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_ORDER = (
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
SLICE_ORDER = (
    "longbench_hotpotqa",
    "longbench_2wikimqa",
    "longbench_musique",
    "overall",
)
SLICE_LABELS = {
    "longbench_hotpotqa": "HotpotQA",
    "longbench_2wikimqa": "2WikiMQA",
    "longbench_musique": "MuSiQue",
    "overall": "Macro pool",
}
COLORS = {
    "base": "#4D4D4D",
    "answer": "#E69F00",
    "evidence_id": "#009E73",
    "evidence": "#0072B2",
}
MARKERS = {"base": "o", "independent": "^", "paired": "s"}


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


def factors(run_name: str) -> tuple[str, str]:
    if run_name == "base":
        return "base", "base"
    pairing, supervision = run_name.split("_", 1)
    return pairing, supervision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    if report.get("schema_version") != "natural-transfer-analysis-v1":
        raise SystemExit("Unexpected natural transfer analysis schema")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render transfer figures") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    intervals = report["run_intervals"]
    matrix = [
        [intervals[slice_name][run_name]["estimate"] for slice_name in SLICE_ORDER]
        for run_name in RUN_ORDER
    ]
    fig, axis = plt.subplots(figsize=(8.3, 5.8))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            axis.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value >= 0.57 else "#202020",
                fontsize=9,
            )
    axis.set_xticks(range(len(SLICE_ORDER)), [SLICE_LABELS[item] for item in SLICE_ORDER])
    axis.set_yticks(range(len(RUN_ORDER)), [RUN_LABELS[item] for item in RUN_ORDER])
    axis.set_xlabel("Natural multi-document benchmark")
    axis.set_title("Natural transfer remains distinct from synthetic position control")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("QA token F1")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    heatmap_files = save_both(fig, args.output_dir, "natural_transfer_heatmap")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.8, 4.9))
    y_values = list(range(len(RUN_ORDER)))
    for y_value, run_name in zip(y_values, RUN_ORDER, strict=True):
        item = intervals["overall"][run_name]
        pairing, supervision = factors(run_name)
        color = COLORS[supervision]
        axis.errorbar(
            item["estimate"],
            y_value,
            xerr=[
                [item["estimate"] - item["ci95_low"]],
                [item["ci95_high"] - item["estimate"]],
            ],
            fmt=MARKERS[pairing],
            color=color,
            fillstyle="full" if pairing in ("base", "paired") else "none",
            capsize=3,
            linewidth=1.4,
            markersize=6.5,
        )
    axis.set_yticks(y_values, [RUN_LABELS[item] for item in RUN_ORDER])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("QA token F1")
    axis.set_title("Natural multi-document transfer with paired-bootstrap 95% intervals")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    interval_files = save_both(fig, args.output_dir, "natural_transfer_intervals")
    plt.close(fig)

    metadata = {
        "schema_version": "natural-transfer-figures-v1",
        "analysis": str(args.analysis.resolve()),
        "analysis_sha256": sha256_file(args.analysis),
        "matplotlib_version": matplotlib.__version__,
        "files": {"heatmap": heatmap_files, "intervals": interval_files},
        "accessibility": {
            "heatmap_alt": (
                "A seven-by-four annotated matrix reports QA token F1 for the base model "
                "and six matched training variants on HotpotQA, 2WikiMQA, MuSiQue, and "
                "their pooled score. Every cell includes its numeric percentage."
            ),
            "intervals_alt": (
                "A dot-and-interval chart compares pooled natural QA token F1 for all "
                "seven conditions using paired task-stratified bootstrap 95% intervals."
            ),
            "table_alternative": "transfer_summary.csv",
        },
    }
    (args.output_dir / "figures.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote natural-transfer publication figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
