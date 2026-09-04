#!/usr/bin/env python3
"""Estimate resumable evaluation progress and cost from saved rows and the measured log rate."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path


RATE_PATTERN = re.compile(r"run=(\S+) saved=(\d+)/(\d+) sec/sample=([0-9.]+)")


def last_rate(log: Path) -> dict[str, object]:
    match = None
    with log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            candidate = RATE_PATTERN.search(line)
            if candidate:
                match = candidate
    if match is None:
        raise ValueError(f"No measured sec/sample line in {log}")
    return {
        "run": match.group(1),
        "saved_this_invocation": int(match.group(2)),
        "selected_this_invocation": int(match.group(3)),
        "seconds_per_sample": float(match.group(4)),
    }


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--runs", required=True, help="Comma-separated canonical result basenames")
    parser.add_argument("--rows-per-run", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = [name.strip() for name in args.runs.split(",") if name.strip()]
    if not runs or args.rows_per_run <= 0:
        raise SystemExit("At least one run and positive --rows-per-run are required")
    try:
        measured = last_rate(args.log)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    counts = {name: count_jsonl(args.result_dir / f"{name}.jsonl") for name in runs}
    if any(count > args.rows_per_run for count in counts.values()):
        raise SystemExit(f"A result exceeds the registered row count: {counts}")
    total = args.rows_per_run * len(runs)
    completed = sum(counts.values())
    remaining = total - completed
    seconds = remaining * float(measured["seconds_per_sample"])
    now = datetime.now().astimezone()
    payload = {
        "schema_version": "evaluation-progress-estimate-v1",
        "measured_at": now.isoformat(),
        "runs": runs,
        "rows_per_run": args.rows_per_run,
        "counts": counts,
        "completed_rows": completed,
        "total_rows": total,
        "completion_fraction": completed / total,
        "remaining_rows": remaining,
        "rate_source": measured,
        "remaining_seconds": seconds,
        "remaining_hours": seconds / 3600,
        "eta": (now + timedelta(seconds=seconds)).isoformat(),
        "estimated_remaining_cost": (
            seconds / 3600 * args.hourly_rate if args.hourly_rate is not None else None
        ),
        "assumption": "latest measured run-level sec/sample remains representative",
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
