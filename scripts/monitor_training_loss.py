#!/usr/bin/env python3
"""Record checkpoint-level loss health without loading or modifying the model."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VARIANTS = (
    "paired_evidence",
    "paired_answer",
    "independent_evidence",
    "independent_answer",
)
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_state(output_root: Path) -> tuple[str, int, Path] | None:
    candidates: list[tuple[float, str, int, Path]] = []
    for variant in VARIANTS:
        if (output_root / variant / "TRAINING_COMPLETE.json").is_file():
            continue
        for checkpoint in (output_root / variant).glob("checkpoint-*"):
            match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
            state = checkpoint / "trainer_state.json"
            if match and state.is_file():
                candidates.append(
                    (state.stat().st_mtime, variant, int(match.group(1)), state)
                )
    if not candidates:
        return None
    _, variant, step, path = max(candidates)
    return variant, step, path


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * probability))]


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) is not None]


def health_record(variant: str, checkpoint_step: int, path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in state.get("log_history", [])
        if "step" in row and "loss" in row
    ]
    if not rows:
        raise ValueError(f"No per-step loss records in {path}")
    last20 = rows[-20:]
    last100 = rows[-100:]
    loss20 = numeric_values(last20, "loss")
    loss100 = numeric_values(last100, "loss")
    gradients = numeric_values(last100, "grad_norm")
    accuracies = numeric_values(last100, "mean_token_accuracy")
    finite = all(
        math.isfinite(float(row[field]))
        for row in last100
        for field in ("loss", "grad_norm", "learning_rate")
        if row.get(field) is not None
    )
    loss_median = statistics.median(loss100)
    accuracy_median = statistics.median(accuracies) if accuracies else None
    alerts: list[str] = []
    severity = "ok"
    if not finite:
        alerts.append("non_finite_metric")
        severity = "critical"
    if gradients and max(gradients) > 100:
        alerts.append("gradient_norm_above_100")
        severity = "warning" if severity == "ok" else severity
    if loss_median < 1e-4 and accuracy_median is not None and accuracy_median > 0.999:
        alerts.append("training_distribution_saturated_check_generalization")
        severity = "warning" if severity == "ok" else severity
    if loss_median > 0 and max(loss100) / loss_median > 1000:
        alerts.append("loss_outlier_above_1000x_median")
        severity = "warning" if severity == "ok" else severity
    last = rows[-1]
    return {
        "schema_version": "training-loss-health-v1",
        "timestamp_utc": utc_now(),
        "variant": variant,
        "checkpoint_step": checkpoint_step,
        "trainer_global_step": int(state.get("global_step", 0)),
        "epoch": float(state.get("epoch", 0.0)),
        "metric_rows": len(rows),
        "loss_last": float(last["loss"]),
        "loss_mean_last_20": statistics.fmean(loss20),
        "loss_median_last_100": loss_median,
        "loss_p95_last_100": quantile(loss100, 0.95),
        "loss_max_last_100": max(loss100),
        "grad_norm_last": float(last["grad_norm"]) if last.get("grad_norm") is not None else None,
        "grad_norm_median_last_100": statistics.median(gradients) if gradients else None,
        "learning_rate_last": (
            float(last["learning_rate"]) if last.get("learning_rate") is not None else None
        ),
        "token_accuracy_median_last_100": accuracy_median,
        "all_recent_metrics_finite": finite,
        "severity": severity,
        "alerts": alerts,
        "source_trainer_state": str(path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.interval_seconds < 5:
        raise SystemExit("--interval-seconds must be at least 5")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stopping = False

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    last_identity: tuple[str, int] | None = None
    while not stopping:
        latest = latest_state(args.output_root)
        if latest is not None:
            variant, step, path = latest
            identity = (variant, step)
            if identity != last_identity:
                record = health_record(variant, step, path)
                with args.output.open("a", encoding="utf-8", buffering=1) as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
                last_identity = identity
        if args.status_file.is_file():
            break
        deadline = time.monotonic() + args.interval_seconds
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
