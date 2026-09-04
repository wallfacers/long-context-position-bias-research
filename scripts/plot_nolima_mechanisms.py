#!/usr/bin/env python3
"""Render publication plots for the NoLiMa retrieval/oracle decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_ORDER = (
    "base",
    "independent_answer",
    "independent_evidence",
    "paired_answer",
    "paired_evidence",
)
RUN_LABELS = {
    "base": "Base",
    "independent_answer": "Independent + answer",
    "independent_evidence": "Independent + evidence",
    "paired_answer": "Paired + answer",
    "paired_evidence": "Paired + evidence",
}
COLORS = {
    "base": "#4D4D4D",
    "independent_answer": "#E69F00",
    "independent_evidence": "#0072B2",
    "paired_answer": "#D55E00",
    "paired_evidence": "#009E73",
}
MARKERS = {
    "base": "o",
    "independent_answer": "^",
    "independent_evidence": "v",
    "paired_answer": "s",
    "paired_evidence": "D",
}
LINESTYLES = {
    "base": ":",
    "independent_answer": "--",
    "independent_evidence": "--",
    "paired_answer": "-",
    "paired_evidence": "-",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_formats(figure: Any, output_dir: Path, stem: str) -> dict[str, str]:
    files = {}
    for extension in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"dpi": 300} if extension == "png" else {}
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        files[extension] = path.name
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    if report.get("schema_version") != "nolima-mechanism-analysis-v1":
        raise SystemExit("Unexpected NoLiMa mechanism schema")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise SystemExit("matplotlib is required") from exc
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals = report["run_intervals"]
    panels = (
        (
            "Answer pathway",
            ("free_answer", "oracle_long_answer", "oracle_short_answer"),
            ("Free long context", "Oracle + long context", "Oracle only"),
        ),
        (
            "Evidence localization",
            ("free_quote", "locate_quote"),
            ("Free answer task", "Locate-only task"),
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    for axis, (title, metrics, labels) in zip(axes, panels, strict=True):
        x_values = list(range(len(metrics)))
        for run in RUN_ORDER:
            estimates = [intervals[run][metric]["estimate"] for metric in metrics]
            lower = [intervals[run][metric]["ci95_low"] for metric in metrics]
            upper = [intervals[run][metric]["ci95_high"] for metric in metrics]
            axis.fill_between(
                x_values, lower, upper, color=COLORS[run], alpha=0.07, linewidth=0
            )
            axis.plot(
                x_values,
                estimates,
                color=COLORS[run],
                marker=MARKERS[run],
                linestyle=LINESTYLES[run],
                linewidth=1.5,
                markersize=5,
                label=RUN_LABELS[run],
            )
        axis.set_title(title)
        axis.set_xticks(x_values, labels)
        axis.set_ylim(-0.02, 1.02)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
        axis.set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )
    figure.suptitle("NoLiMa mechanism decomposition", y=1.08, fontsize=13)
    figure.text(
        0.5,
        -0.015,
        "Bands are 95% case-cluster bootstrap intervals (10 semantic cases); oracle inputs are deduplicated across source positions.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.91))
    files = save_formats(figure, args.output_dir, "nolima_mechanism_decomposition")
    plt.close(figure)
    metadata = {
        "schema_version": "nolima-mechanism-figures-v1",
        "analysis": str(args.analysis.resolve()),
        "analysis_sha256": sha256_file(args.analysis),
        "matplotlib_version": matplotlib.__version__,
        "files": files,
        "accessibility": {
            "encoding": "Runs use redundant color, line style, and marker encodings.",
            "alt": (
                "Two panels compare free, oracle-long, and oracle-short answer accuracy "
                "and free versus locate-only exact-quote accuracy for base and four trained conditions."
            ),
            "table_alternative": "nolima_mechanism_summary.csv",
        },
    }
    (args.output_dir / "figures.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote NoLiMa mechanism figure to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
