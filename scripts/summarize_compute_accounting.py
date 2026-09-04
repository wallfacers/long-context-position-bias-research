#!/usr/bin/env python3
"""Summarize active training/evaluation GPU time from audited final-run metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_id(kind: str, payload: dict[str, Any]) -> str:
    if kind == "training":
        identity = {
            "kind": kind,
            "schema_version": payload.get("schema_version"),
            "run_id": payload.get("run_id"),
            "finished_at": payload.get("finished_at"),
            "global_step": payload.get("global_step"),
        }
    else:
        identity = {
            "kind": kind,
            "schema_version": payload.get("schema_version"),
            "started_at": payload.get("started_at"),
            "last_finished_at": payload.get("last_finished_at"),
            "model": payload.get("model"),
            "adapter_sha256": payload.get("adapter_sha256"),
            "data_sha256": payload.get("data_sha256"),
            "selection_sha256": payload.get("selection_sha256"),
            "selected_samples": payload.get("selected_samples"),
        }
    rendered = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Accounting input is outside --project-root: {path}") from exc


def collect(
    project_root: Path,
    training_roots: list[Path],
    eval_roots: list[Path],
    expected_training_step: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    inputs = [("training", root) for root in training_roots] + [
        ("evaluation", root) for root in eval_roots
    ]
    for kind, root in inputs:
        resolved_root = root.resolve()
        relative_path(resolved_root, project_root)
        if not resolved_root.is_dir():
            raise ValueError(f"Accounting root is missing: {root}")
        pattern = "CANARY_COMPLETE.json" if kind == "training" else "*.run.json"
        paths = sorted(resolved_root.rglob(pattern))
        if not paths:
            raise ValueError(f"Accounting root has no {pattern} files: {root}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if kind == "training":
                if (
                    payload.get("schema_version") != "qlora-result-v1"
                    or int(payload.get("global_step", -1)) != expected_training_step
                ):
                    raise ValueError(f"Invalid completed training metadata: {path}")
                seconds = float(payload.get("elapsed_seconds_this_invocation", 0.0))
                units = int(payload["global_step"])
            else:
                if (
                    payload.get("schema_version")
                    not in {"vllm-eval-run-v1", "ifeval-vllm-run-v1"}
                    or payload.get("status") != "selection_complete"
                ):
                    raise ValueError(f"Invalid completed evaluation metadata: {path}")
                seconds = float(payload.get("elapsed_seconds_total", 0.0))
                units = int(
                    payload.get("selected_samples", payload.get("samples", 0))
                )
            if seconds <= 0 or units <= 0:
                raise ValueError(f"Non-positive accounting duration or units: {path}")
            identifier = event_id(kind, payload)
            record = {
                "event_id": identifier,
                "kind": kind,
                "seconds": seconds,
                "units": units,
                "source": relative_path(path, project_root),
                "source_sha256": sha256_file(path),
            }
            previous = records.get(identifier)
            if previous is None:
                records[identifier] = record
            else:
                if (
                    previous["kind"] != kind
                    or abs(float(previous["seconds"]) - seconds) > 1e-9
                    or int(previous["units"]) != units
                    or previous["source_sha256"] != record["source_sha256"]
                ):
                    raise ValueError(f"Conflicting copies of accounting event {identifier}")
                duplicates.append(
                    {
                        "event_id": identifier,
                        "kept": previous["source"],
                        "deduplicated": record["source"],
                    }
                )
    return sorted(records.values(), key=lambda item: item["event_id"]), duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, action="append", default=[])
    parser.add_argument("--eval-root", type=Path, action="append", default=[])
    parser.add_argument("--expected-training-step", type=int, default=100)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hourly_rate <= 0:
        raise SystemExit("--hourly-rate must be positive")
    if args.expected_training_step <= 0:
        raise SystemExit("--expected-training-step must be positive")
    if not args.training_root and not args.eval_root:
        raise SystemExit("At least one --training-root or --eval-root is required")
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise SystemExit("--project-root must be an existing directory")
    try:
        records, duplicates = collect(
            project_root,
            args.training_root,
            args.eval_root,
            args.expected_training_step,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    by_kind = {}
    for kind in ("training", "evaluation"):
        selected = [record for record in records if record["kind"] == kind]
        seconds = sum(float(record["seconds"]) for record in selected)
        by_kind[kind] = {
            "events": len(selected),
            "units": sum(int(record["units"]) for record in selected),
            "active_gpu_seconds": seconds,
            "active_gpu_hours": seconds / 3600.0,
        }
    total_seconds = sum(float(record["seconds"]) for record in records)
    report = {
        "schema_version": "active-gpu-compute-accounting-v1",
        "status": "validated",
        "scope": (
            "Completed trainer/vLLM engine time only. This is an auditable active-GPU "
            "lower bound, not a cloud bill; model loading, CPU analysis, packaging, "
            "idle allocation, and provider billing granularity are excluded."
        ),
        "hourly_rate_cny": args.hourly_rate,
        "expected_training_step": args.expected_training_step,
        "input_roots": {
            "training": [
                relative_path(path.resolve(), project_root)
                for path in args.training_root
            ],
            "evaluation": [
                relative_path(path.resolve(), project_root)
                for path in args.eval_root
            ],
        },
        "by_kind": by_kind,
        "total_active_gpu_seconds": total_seconds,
        "total_active_gpu_hours": total_seconds / 3600.0,
        "active_gpu_cost_lower_bound_cny": total_seconds / 3600.0 * args.hourly_rate,
        "unique_events": len(records),
        "deduplicated_event_copies": len(duplicates),
        "events": records,
        "duplicates": duplicates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Active GPU accounting: "
        f"events={len(records)} hours={report['total_active_gpu_hours']:.3f} "
        f"cost_lower_bound=¥{report['active_gpu_cost_lower_bound_cny']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
