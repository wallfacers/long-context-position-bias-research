#!/usr/bin/env python3
"""Sample GPU telemetry and checkpoint progress without touching training state."""

from __future__ import annotations

import argparse
import csv
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = (
    "paired_evidence",
    "paired_answer",
    "independent_evidence",
    "independent_answer",
)
GPU_FIELDS = (
    "gpu_name",
    "driver_version",
    "gpu_utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "power_limit_w",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "pstate",
)
CSV_FIELDS = (
    "timestamp_utc",
    "active_variant",
    "latest_checkpoint_step",
    *GPU_FIELDS,
    "nvidia_smi_returncode",
)
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def active_variant(log_path: Path) -> str:
    if not log_path.is_file():
        return "unknown"
    active = "between_variants"
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("BEGIN "):
            parts = line.split()
            if len(parts) >= 2:
                active = parts[1]
        elif line.startswith("DONE ") or line.startswith("FAILED "):
            parts = line.split()
            if len(parts) >= 2 and parts[1] == active:
                active = "between_variants"
    return active


def latest_checkpoint(output_root: Path, variant: str) -> int:
    if variant not in VARIANTS:
        return 0
    steps: list[int] = []
    for path in (output_root / variant).glob("checkpoint-*"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match and (path / "trainer_state.json").is_file():
            steps.append(int(match.group(1)))
    return max(steps, default=0)


def gpu_sample() -> tuple[int, list[str]]:
    query = ",".join(
        (
            "name",
            "driver_version",
            "utilization.gpu",
            "memory.used",
            "memory.total",
            "power.draw",
            "power.limit",
            "temperature.gpu",
            "clocks.sm",
            "clocks.mem",
            "pstate",
        )
    )
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=15,
    )
    values = [item.strip() for item in result.stdout.strip().split(",")]
    if result.returncode or len(values) != len(GPU_FIELDS):
        values = [""] * len(GPU_FIELDS)
    return result.returncode, values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-log", type=Path, required=True)
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
    write_header = not args.output.is_file() or args.output.stat().st_size == 0
    with args.output.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        while not stopping:
            variant = active_variant(args.queue_log)
            returncode, values = gpu_sample()
            row = {
                "timestamp_utc": utc_now(),
                "active_variant": variant,
                "latest_checkpoint_step": latest_checkpoint(args.output_root, variant),
                **dict(zip(GPU_FIELDS, values, strict=True)),
                "nvidia_smi_returncode": returncode,
            }
            writer.writerow(row)
            handle.flush()
            if args.status_file.is_file():
                break
            deadline = time.monotonic() + args.interval_seconds
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
