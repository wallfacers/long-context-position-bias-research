#!/usr/bin/env python3
"""Build a task-window cost ledger for the AutoDL paper pilot."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


LOG_TIME = re.compile(r"(\d{8}T\d{6}Z)")


def parse_time(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def log_start(path: Path) -> datetime:
    match = LOG_TIME.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse UTC timestamp from {path}")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )


def status_finish(path: Path) -> datetime:
    match = re.search(r"finished_at=([^\s]+)", path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Cannot parse finished_at from {path}")
    return parse_time(match.group(1))


def entry(
    name: str,
    *,
    start: datetime,
    finish: datetime,
    hourly_rate_cny: float,
    source: str,
    status: str,
    note: str,
) -> dict[str, Any]:
    seconds = (finish - start).total_seconds()
    if seconds <= 0:
        raise ValueError(f"Non-positive task window for {name}: {start} -> {finish}")
    return {
        "name": name,
        "status": status,
        "started_at": start.isoformat(),
        "finished_at": finish.isoformat(),
        "wall_seconds": seconds,
        "wall_hours": seconds / 3600,
        "hourly_rate_cny": hourly_rate_cny,
        "estimated_cost_cny": seconds / 3600 * hourly_rate_cny,
        "source": source,
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--hourly-rate-cny", type=float, default=2.78)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    rate = args.hourly_rate_cny
    if rate <= 0:
        raise SystemExit("--hourly-rate-cny must be positive")

    entries: list[dict[str, Any]] = []
    training_logs = sorted((root / "outputs").glob("*/logs/*.log"))
    completions = sorted((root / "outputs").glob("*/TRAINING_COMPLETE.json"))
    if not training_logs or len(completions) != 4:
        raise SystemExit("Training logs or four completion records are missing")
    training_start = min(log_start(path) for path in training_logs)
    training_finish = max(
        parse_time(json.loads(path.read_text(encoding="utf-8"))["finished_at"])
        for path in completions
    )
    entries.append(
        entry(
            "four_variant_qlora_training_window",
            start=training_start,
            finish=training_finish,
            hourly_rate_cny=rate,
            source="outputs/*/logs + outputs/*/TRAINING_COMPLETE.json",
            status="completed",
            note=(
                "Recorded wall window from the first QLoRA invocation through the last "
                "variant completion; includes canary/resume and between-run orchestration."
            ),
        )
    )

    gate_dir = root / "results/dev_gate"
    gate_logs = sorted(gate_dir.glob("logs/gate-*.log"), key=log_start)
    gate_report = json.loads((gate_dir / "gate-report.json").read_text(encoding="utf-8"))
    if not gate_logs or gate_report.get("status") != "validated":
        raise SystemExit("Validated dev gate artifacts are missing")
    for index, path in enumerate(gate_logs[:-1], start=1):
        start = log_start(path)
        finish = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        entries.append(
            entry(
                f"dev_gate_failed_attempt_{index}",
                start=start,
                finish=finish,
                hourly_rate_cny=rate,
                source=str(path.relative_to(root)),
                status="failed_preserved",
                note="Failed compatibility attempt retained in the cost ledger.",
            )
        )
    successful_gate_finish = parse_time(gate_report["created_at"])
    successful_gate_seconds = float(gate_report["gate"]["wall_seconds"])
    entries.append(
        entry(
            "dev_gate_validated",
            start=successful_gate_finish
            - timedelta(seconds=successful_gate_seconds),
            finish=successful_gate_finish,
            hourly_rate_cny=rate,
            source="results/dev_gate/gate-report.json",
            status="validated",
            note="Validated 100-prediction execution gate.",
        )
    )

    diagnostics = (
        (
            "full_eval_cap128_protocol_diagnostic",
            root / "results/test_full_cap128_diagnostic",
            args.external_root / "position-bias-test-full-cap128-diagnostic.status",
            "Stopped after the balanced prefix exposed valid JSON truncation.",
        ),
        (
            "full_eval_cap256_window_preflight",
            root / "results/test_full_cap256_window_preflight",
            args.external_root / "position-bias-test-full-cap256-window-preflight.status",
            "Stopped after checking that 256 tokens exceeded the longest 32K headroom.",
        ),
    )
    for name, directory, status_path, note in diagnostics:
        logs = sorted(directory.glob("logs/full-*.log"), key=log_start)
        if len(logs) != 1 or not status_path.is_file():
            raise SystemExit(f"Missing diagnostic timing evidence for {name}")
        entries.append(
            entry(
                name,
                start=log_start(logs[0]),
                finish=status_finish(status_path),
                hourly_rate_cny=rate,
                source=f"{logs[0].relative_to(root)} + {status_path}",
                status="stopped_and_preserved",
                note=note,
            )
        )

    full_validation_path = root / "results/test_full/validation-report.json"
    validation = json.loads(full_validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "validated":
        raise SystemExit("Full evaluation validation is missing")
    full_log = max((root / "results/test_full/logs").glob("full-*.log"), key=log_start)
    full_start = log_start(full_log)
    full_seconds = float(validation["execution"]["wall_seconds"])
    entries.append(
        entry(
            "full_eval_cap176_validated",
            start=full_start,
            finish=full_start + timedelta(seconds=full_seconds),
            hourly_rate_cny=rate,
            source="results/test_full/validation-report.json",
            status="validated",
            note="Formal five-run, 21,000-prediction test wall window.",
        )
    )

    total_seconds = sum(item["wall_seconds"] for item in entries)
    total_cost = sum(item["estimated_cost_cny"] for item in entries)
    payload = {
        "schema_version": "autodl-pilot-cost-ledger-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "currency": "CNY",
        "hourly_rate_cny": rate,
        "entries": entries,
        "totals": {
            "recorded_task_wall_seconds": total_seconds,
            "recorded_task_wall_hours": total_seconds / 3600,
            "estimated_recorded_task_cost_cny": total_cost,
        },
        "accounting_boundary": (
            "Task-window estimate, not the AutoDL invoice. It includes recorded training, "
            "gate, protocol-diagnostic, and formal-evaluation windows; it excludes unrecorded "
            "instance idle time, storage, transfer, and post-processing after validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
