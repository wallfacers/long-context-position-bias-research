#!/usr/bin/env python3
"""Persist evaluation progress and GPU telemetry without modifying the job."""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


RUN_NAMES = (
    "base",
    "paired_evidence",
    "paired_answer",
    "independent_evidence",
    "independent_answer",
)
GPU_QUERY = (
    "name,driver_version,utilization.gpu,memory.used,memory.total,power.draw,"
    "power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate"
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
    *(f"{name}_rows" for name in RUN_NAMES),
    "total_rows",
    "expected_rows",
    "progress_percent",
    *GPU_FIELDS,
    "nvidia_smi_returncode",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def gpu_sample() -> tuple[int, list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-per-run", type=int, default=4200)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.expected_per_run <= 0 or args.interval_seconds < 5:
        raise SystemExit("Expected rows must be positive and interval must be at least 5 seconds")

    stopping = False

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    expected = args.expected_per_run * len(RUN_NAMES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.is_file() or args.output.stat().st_size == 0
    with args.output.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        while not stopping:
            counts = {
                f"{name}_rows": line_count(args.results_dir / f"{name}.jsonl")
                for name in RUN_NAMES
            }
            total = sum(counts.values())
            returncode, values = gpu_sample()
            writer.writerow(
                {
                    "timestamp_utc": utc_now(),
                    **counts,
                    "total_rows": total,
                    "expected_rows": expected,
                    "progress_percent": total / expected * 100,
                    **dict(zip(GPU_FIELDS, values, strict=True)),
                    "nvidia_smi_returncode": returncode,
                }
            )
            handle.flush()
            if args.status_file.is_file():
                break
            deadline = time.monotonic() + args.interval_seconds
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
