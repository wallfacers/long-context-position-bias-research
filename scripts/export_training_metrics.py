#!/usr/bin/env python3
"""Export complete Trainer histories into portable, auditable metric files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
DEFAULT_VARIANTS = (
    "paired_evidence",
    "paired_evidence_id",
    "paired_answer",
    "independent_evidence",
    "independent_evidence_id",
    "independent_answer",
)
METRIC_COLUMNS = (
    "variant",
    "step",
    "epoch",
    "loss",
    "grad_norm",
    "learning_rate",
    "entropy",
    "num_tokens",
    "mean_token_accuracy",
    "eval_loss",
)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_checkpoint(output_dir: Path) -> tuple[int, Path]:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match and (path / "trainer_state.json").is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError(f"No checkpoint with trainer_state.json in {output_dir}")
    return max(candidates)


def finite_number(value: Any, *, field: str, step: int) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} at step {step} is not numeric: {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} at step {step} is not finite: {value!r}")
    return value


def metric_rows(state: dict[str, Any], variant: str) -> tuple[list[dict[str, Any]], int]:
    by_step: dict[int, dict[str, Any]] = {}
    duplicates = 0
    for raw in state.get("log_history", []):
        # Trainer also logs a final runtime summary at the last step. It may
        # contain ``epoch`` but is not a per-step optimizer record and must not
        # overwrite the actual last-step loss after checkpoint recovery.
        if "step" not in raw or not any(
            key in raw for key in ("loss", "eval_loss", "learning_rate")
        ):
            continue
        step = int(raw["step"])
        row: dict[str, Any] = {"variant": variant, "step": step}
        for field in METRIC_COLUMNS[2:]:
            row[field] = finite_number(raw.get(field), field=field, step=step)
        if step in by_step:
            duplicates += 1
        by_step[step] = row
    return [by_step[step] for step in sorted(by_step)], duplicates


def median_tail(rows: list[dict[str, Any]], key: str, count: int = 100) -> float | None:
    values = [float(row[key]) for row in rows[-count:] if row.get(key) is not None]
    return statistics.median(values) if values else None


def summarize(
    *,
    variant: str,
    rows: list[dict[str, Any]],
    duplicate_steps: int,
    state: dict[str, Any],
    checkpoint_step: int,
    expected_steps: int,
    output_dir: Path,
    completion_record_name: str = "TRAINING_COMPLETE.json",
) -> dict[str, Any]:
    steps = [int(row["step"]) for row in rows]
    expected = set(range(1, max(steps, default=0) + 1))
    missing = sorted(expected - set(steps))
    loss_rows = [row for row in rows if row.get("loss") is not None]
    accuracy_rows = [row for row in rows if row.get("mean_token_accuracy") is not None]
    warnings: list[str] = []
    if missing:
        warnings.append(f"missing {len(missing)} logged steps")
    if duplicate_steps:
        warnings.append(f"deduplicated {duplicate_steps} repeated steps")
    if loss_rows and median_tail(loss_rows, "loss") is not None:
        if median_tail(loss_rows, "loss") < 1e-4:
            warnings.append("last-100 median training loss is below 1e-4; verify generalization")
    if accuracy_rows and median_tail(accuracy_rows, "mean_token_accuracy") is not None:
        if median_tail(accuracy_rows, "mean_token_accuracy") > 0.999:
            warnings.append("last-100 median token accuracy exceeds 0.999; verify overfitting")
    minimum_loss_row = min(loss_rows, key=lambda row: float(row["loss"])) if loss_rows else None
    completion_path = output_dir / completion_record_name
    completion = (
        json.loads(completion_path.read_text(encoding="utf-8"))
        if completion_path.is_file()
        else None
    )
    return {
        "schema_version": "training-metrics-summary-v1",
        "variant": variant,
        "checkpoint_step": checkpoint_step,
        "trainer_state_global_step": int(state.get("global_step", 0)),
        "expected_steps": expected_steps,
        "training_complete": completion is not None,
        "completion_record_name": completion_record_name,
        "completion_record": completion,
        "metric_rows": len(rows),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "missing_steps": missing,
        "duplicate_steps": duplicate_steps,
        "first_loss": float(loss_rows[0]["loss"]) if loss_rows else None,
        "final_loss": float(loss_rows[-1]["loss"]) if loss_rows else None,
        "minimum_loss": float(minimum_loss_row["loss"]) if minimum_loss_row else None,
        "minimum_loss_step": int(minimum_loss_row["step"]) if minimum_loss_row else None,
        "median_loss_last_100": median_tail(loss_rows, "loss"),
        "final_learning_rate": (
            float(rows[-1]["learning_rate"])
            if rows and rows[-1].get("learning_rate") is not None
            else None
        ),
        "final_mean_token_accuracy": (
            float(accuracy_rows[-1]["mean_token_accuracy"]) if accuracy_rows else None
        ),
        "median_mean_token_accuracy_last_100": median_tail(
            accuracy_rows, "mean_token_accuracy"
        ),
        "warnings": warnings,
    }


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--expected-steps", type=int, default=2000)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--completion-record-name",
        choices=("TRAINING_COMPLETE.json", "CANARY_COMPLETE.json"),
        default="TRAINING_COMPLETE.json",
        help="Completion gate required by --require-complete.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    variants = tuple(args.variants or DEFAULT_VARIANTS)
    metrics_dir = args.diagnostics_dir / "metrics"
    state_dir = args.diagnostics_dir / "raw_trainer_state"
    summaries: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    errors: list[str] = []
    for variant in variants:
        output_dir = args.output_root / variant
        try:
            checkpoint_step, checkpoint = latest_checkpoint(output_dir)
            state_path = checkpoint / "trainer_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            rows, duplicates = metric_rows(state, variant)
            if not rows:
                raise ValueError(f"No step metrics in {state_path}")
            summary = summarize(
                variant=variant,
                rows=rows,
                duplicate_steps=duplicates,
                state=state,
                checkpoint_step=checkpoint_step,
                expected_steps=args.expected_steps,
                output_dir=output_dir,
                completion_record_name=args.completion_record_name,
            )
            if args.require_complete:
                if not summary["training_complete"]:
                    raise ValueError(
                        f"Missing {output_dir / args.completion_record_name}"
                    )
                if summary["last_step"] != args.expected_steps:
                    raise ValueError(
                        f"{variant} has metrics through {summary['last_step']}, "
                        f"expected {args.expected_steps}"
                    )
                if summary["missing_steps"]:
                    raise ValueError(f"{variant} has missing logged steps")
            write_csv_atomic(metrics_dir / f"{variant}.csv", rows)
            write_jsonl_atomic(metrics_dir / f"{variant}.jsonl", rows)
            state_dir.mkdir(parents=True, exist_ok=True)
            state_copy = state_dir / f"{variant}.trainer_state.json"
            temporary = state_copy.with_name(state_copy.name + f".tmp-{os.getpid()}")
            shutil.copyfile(state_path, temporary)
            temporary.replace(state_copy)
            summary["source_trainer_state"] = str(state_path.resolve())
            summary["source_trainer_state_sha256"] = sha256_file(state_path)
            summary["exported_trainer_state_sha256"] = sha256_file(state_copy)
            summaries.append(summary)
            combined.extend(rows)
            print(
                f"{variant}: exported {len(rows)} steps from checkpoint-{checkpoint_step}",
                flush=True,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{variant}: {exc}")
            if args.require_complete:
                raise SystemExit("METRIC EXPORT FAILED: " + "; ".join(errors)) from exc
            print(f"SKIP {variant}: {exc}", flush=True)

    if not combined:
        raise SystemExit("METRIC EXPORT FAILED: no metrics were exported")
    write_csv_atomic(args.diagnostics_dir / "training_metrics.csv", combined)
    write_jsonl_atomic(args.diagnostics_dir / "training_metrics.jsonl", combined)
    report = {
        "schema_version": "training-metrics-export-v1",
        "expected_steps_per_variant": args.expected_steps,
        "require_complete": args.require_complete,
        "variants_requested": list(variants),
        "variants_exported": [summary["variant"] for summary in summaries],
        "total_metric_rows": len(combined),
        "summaries": summaries,
        "nonfatal_errors": errors,
    }
    write_json_atomic(args.diagnostics_dir / "training_metrics_summary.json", report)
    print(f"Wrote publication metrics to {args.diagnostics_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
