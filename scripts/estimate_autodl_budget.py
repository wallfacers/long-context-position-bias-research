#!/usr/bin/env python3
"""Estimate AutoDL spend from measured or conservative workload throughput.

The estimator deliberately separates GPU-hours from wall-clock hours. Running
independent ablations in parallel finishes sooner but does not reduce the total
per-GPU bill.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "autodl_pilot_budget.json"
)


@dataclass(frozen=True)
class WorkloadEstimate:
    name: str
    gpu: str
    runs: int
    parallel_instances: int
    units_per_run: int
    seconds_per_unit: float
    fixed_hours_per_run: float
    gpu_hours: float
    wall_hours: float
    hourly_rate: float
    cost: float


@dataclass(frozen=True)
class BudgetEstimate:
    workloads: tuple[WorkloadEstimate, ...]
    total_gpu_hours: float
    single_queue_wall_hours: float
    gpu_cost: float
    one_time_cost: float
    expected_total: float
    contingency_ratio: float
    budget_ceiling: float
    round_up_to: float
    recommended_top_up: float


def parse_start_time(value: str) -> datetime:
    """Parse an explicit timezone-aware ISO-8601 scheduling baseline."""
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"--start-time must be ISO-8601: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--start-time must include a UTC offset or Z")
    return parsed


def completion_times(
    estimate: BudgetEstimate, start_time: datetime
) -> tuple[datetime, datetime]:
    """Return single-queue expected and contingency completion times."""
    expected = start_time + timedelta(hours=estimate.single_queue_wall_hours)
    upper = start_time + timedelta(
        hours=estimate.single_queue_wall_hours * (1.0 + estimate.contingency_ratio)
    )
    return expected, upper


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "name",
        "currency",
        "contingency_ratio",
        "round_up_to",
        "pricing",
        "workloads",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Config is missing keys: {', '.join(sorted(missing))}")
    if config["contingency_ratio"] < 0:
        raise ValueError("contingency_ratio must be non-negative")
    if config["round_up_to"] <= 0:
        raise ValueError("round_up_to must be positive")

    pricing = config["pricing"]
    names: set[str] = set()
    for workload in config["workloads"]:
        name = workload["name"]
        if name in names:
            raise ValueError(f"Duplicate workload name: {name}")
        names.add(name)
        if workload["gpu"] not in pricing:
            raise ValueError(f"No price configured for GPU: {workload['gpu']}")
        for key in ("runs", "parallel_instances", "units_per_run"):
            if int(workload[key]) <= 0:
                raise ValueError(f"{name}.{key} must be positive")
        for key in ("seconds_per_unit", "fixed_hours_per_run"):
            if float(workload[key]) < 0:
                raise ValueError(f"{name}.{key} must be non-negative")


def parse_assignments(values: Iterable[str], option: str) -> dict[str, float]:
    assignments: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects TARGET=VALUE, got: {value}")
        target, raw_number = value.split("=", 1)
        target = target.strip()
        if not target:
            raise ValueError(f"{option} target cannot be empty")
        try:
            number = float(raw_number)
        except ValueError as exc:
            raise ValueError(f"{option} value must be numeric: {value}") from exc
        if number < 0:
            raise ValueError(f"{option} value must be non-negative: {value}")
        assignments[target] = number
    return assignments


def measured_seconds(path: Path, key: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} lacks a valid {key}") from exc
    if value <= 0:
        raise ValueError(f"{path}.{key} must be positive")
    return value


def apply_overrides(
    config: dict[str, Any],
    rate_overrides: dict[str, float],
    seconds_overrides: dict[str, float],
) -> dict[str, Any]:
    updated = copy.deepcopy(config)

    unknown_gpus = rate_overrides.keys() - updated["pricing"].keys()
    if unknown_gpus:
        raise ValueError(f"Unknown GPU override: {', '.join(sorted(unknown_gpus))}")
    for gpu, rate in rate_overrides.items():
        updated["pricing"][gpu]["hourly_rate"] = rate

    matched_targets: set[str] = set()
    for workload in updated["workloads"]:
        targets = {workload["name"]}
        if workload.get("calibration_group"):
            targets.add(workload["calibration_group"])
        for target, seconds in seconds_overrides.items():
            if target in targets:
                workload["seconds_per_unit"] = seconds
                matched_targets.add(target)

    unmatched = seconds_overrides.keys() - matched_targets
    if unmatched:
        raise ValueError(
            "Unknown workload/calibration group: " + ", ".join(sorted(unmatched))
        )
    return updated


def apply_remaining_units(
    config: dict[str, Any], remaining_units: dict[str, float]
) -> dict[str, Any]:
    """Collapse in-progress workloads to their measured total residual units.

    The residual estimate deliberately clears per-run fixed overhead because that
    overhead has already been paid for a running workload. Completed workloads
    should still use ``--skip`` instead.
    """
    updated = copy.deepcopy(config)
    known = {workload["name"] for workload in updated["workloads"]}
    unknown = remaining_units.keys() - known
    if unknown:
        raise ValueError(f"Unknown remaining-unit workload: {', '.join(sorted(unknown))}")
    for name, raw_units in remaining_units.items():
        if raw_units <= 0 or int(raw_units) != raw_units:
            raise ValueError(f"Remaining units must be a positive integer: {name}={raw_units}")
        workload = next(item for item in updated["workloads"] if item["name"] == name)
        workload["runs"] = 1
        workload["parallel_instances"] = 1
        workload["units_per_run"] = int(raw_units)
        workload["fixed_hours_per_run"] = 0.0
        workload["notes"] = (
            f"Residual snapshot: {int(raw_units)} total units remain; already-paid "
            "per-run fixed overhead is excluded."
        )
    validate_config(updated)
    return updated


def estimate_budget(
    config: dict[str, Any],
    *,
    skipped_workloads: Iterable[str] = (),
) -> BudgetEstimate:
    skipped = set(skipped_workloads)
    known_names = {workload["name"] for workload in config["workloads"]}
    unknown_skips = skipped - known_names
    if unknown_skips:
        raise ValueError(f"Unknown skipped workload: {', '.join(sorted(unknown_skips))}")

    estimates: list[WorkloadEstimate] = []
    for workload in config["workloads"]:
        if workload["name"] in skipped:
            continue

        runs = int(workload["runs"])
        parallel = min(int(workload["parallel_instances"]), runs)
        units = int(workload["units_per_run"])
        seconds = float(workload["seconds_per_unit"])
        fixed = float(workload["fixed_hours_per_run"])
        per_run_hours = units * seconds / 3600.0 + fixed
        gpu_hours = runs * per_run_hours
        wall_hours = math.ceil(runs / parallel) * per_run_hours
        rate = float(config["pricing"][workload["gpu"]]["hourly_rate"])
        estimates.append(
            WorkloadEstimate(
                name=workload["name"],
                gpu=workload["gpu"],
                runs=runs,
                parallel_instances=parallel,
                units_per_run=units,
                seconds_per_unit=seconds,
                fixed_hours_per_run=fixed,
                gpu_hours=gpu_hours,
                wall_hours=wall_hours,
                hourly_rate=rate,
                cost=gpu_hours * rate,
            )
        )

    gpu_cost = sum(item.cost for item in estimates)
    one_time_cost = sum(
        float(item["amount"]) for item in config.get("one_time_costs", [])
    )
    expected = gpu_cost + one_time_cost
    contingency = float(config["contingency_ratio"])
    ceiling = expected * (1.0 + contingency)
    rounding = float(config["round_up_to"])
    recommended = math.ceil(ceiling / rounding) * rounding
    return BudgetEstimate(
        workloads=tuple(estimates),
        total_gpu_hours=sum(item.gpu_hours for item in estimates),
        single_queue_wall_hours=sum(item.wall_hours for item in estimates),
        gpu_cost=gpu_cost,
        one_time_cost=one_time_cost,
        expected_total=expected,
        contingency_ratio=contingency,
        budget_ceiling=ceiling,
        round_up_to=rounding,
        recommended_top_up=recommended,
    )


def format_money(value: float, currency: str) -> str:
    symbol = "¥" if currency == "CNY" else f"{currency} "
    return f"{symbol}{value:,.2f}"


def print_report(
    config: dict[str, Any],
    estimate: BudgetEstimate,
    start_time: datetime | None = None,
) -> None:
    currency = config["currency"]
    print(f"Scenario: {config['name']} (rates as of {config.get('as_of', 'unknown')})")
    print()
    header = (
        f"{'workload':43} {'GPU':19} {'runs':>4} {'sec/unit':>8} "
        f"{'GPU-h':>8} {'wall-h':>8} {'cost':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in estimate.workloads:
        print(
            f"{item.name:43} {item.gpu:19} {item.runs:>4} "
            f"{item.seconds_per_unit:>8.2f} {item.gpu_hours:>8.2f} "
            f"{item.wall_hours:>8.2f} {format_money(item.cost, currency):>10}"
        )

    print()
    print(f"Total GPU-hours:     {estimate.total_gpu_hours:,.2f}")
    print(f"Single-queue hours:  {estimate.single_queue_wall_hours:,.2f}")
    if start_time is not None:
        expected_eta, upper_eta = completion_times(estimate, start_time)
        print(f"Schedule starts:     {start_time.isoformat()}")
        print(f"Expected finish:     {expected_eta.isoformat()}")
        print(f"Contingency finish:  {upper_eta.isoformat()}")
    print(f"GPU subtotal:       {format_money(estimate.gpu_cost, currency)}")
    print(f"One-time reserve:   {format_money(estimate.one_time_cost, currency)}")
    print(f"Expected spend:     {format_money(estimate.expected_total, currency)}")
    print(
        f"Ceiling (+{estimate.contingency_ratio:.0%}): "
        f"{format_money(estimate.budget_ceiling, currency)}"
    )
    print(
        f"Recommended top-up: {format_money(estimate.recommended_top_up, currency)} "
        f"(rounded up to {format_money(estimate.round_up_to, currency)})"
    )
    print()
    print("Parallel instances reduce wall time only; total GPU-hours and cost stay the same.")


def as_json(
    config: dict[str, Any],
    estimate: BudgetEstimate,
    start_time: datetime | None = None,
) -> str:
    payload = {
        "scenario": config["name"],
        "as_of": config.get("as_of"),
        "currency": config["currency"],
        "workloads": [item.__dict__ for item in estimate.workloads],
        "total_gpu_hours": estimate.total_gpu_hours,
        "single_queue_wall_hours": estimate.single_queue_wall_hours,
        "gpu_cost": estimate.gpu_cost,
        "one_time_cost": estimate.one_time_cost,
        "expected_total": estimate.expected_total,
        "contingency_ratio": estimate.contingency_ratio,
        "budget_ceiling": estimate.budget_ceiling,
        "round_up_to": estimate.round_up_to,
        "recommended_top_up": estimate.recommended_top_up,
    }
    if start_time is not None:
        expected_eta, upper_eta = completion_times(estimate, start_time)
        payload.update(
            {
                "schedule_start": start_time.isoformat(),
                "expected_single_queue_finish": expected_eta.isoformat(),
                "contingency_single_queue_finish": upper_eta.isoformat(),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--rate",
        action="append",
        default=[],
        metavar="GPU=RATE",
        help="Override an hourly GPU rate; repeat as needed.",
    )
    parser.add_argument(
        "--seconds",
        action="append",
        default=[],
        metavar="TARGET=SECONDS",
        help=(
            "Override seconds per unit by workload name or calibration group; "
            "repeat as needed."
        ),
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="WORKLOAD",
        help="Exclude an already completed workload; repeat as needed.",
    )
    parser.add_argument(
        "--remaining-units",
        action="append",
        default=[],
        metavar="WORKLOAD=UNITS",
        help=(
            "Replace an in-progress workload by its measured total remaining units; "
            "clears already-paid fixed overhead. Repeat as needed."
        ),
    )
    parser.add_argument(
        "--train-canary",
        type=Path,
        help="Read qlora_train seconds/step from CANARY_COMPLETE.json.",
    )
    parser.add_argument(
        "--eval-canary",
        type=Path,
        help="Read eval_request seconds/sample from a vLLM .run.json.",
    )
    parser.add_argument(
        "--start-time",
        help=(
            "Timezone-aware ISO-8601 baseline for a sequential single-queue ETA, "
            "for example 2026-08-28T23:42:27+08:00."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        seconds_overrides = parse_assignments(args.seconds, "--seconds")
        if args.train_canary:
            if "qlora_train" in seconds_overrides:
                raise ValueError("Do not combine --train-canary with --seconds qlora_train=...")
            seconds_overrides["qlora_train"] = measured_seconds(
                args.train_canary, "seconds_per_step_this_invocation"
            )
        if args.eval_canary:
            if "eval_request" in seconds_overrides:
                raise ValueError("Do not combine --eval-canary with --seconds eval_request=...")
            seconds_overrides["eval_request"] = measured_seconds(
                args.eval_canary, "seconds_per_sample_this_invocation"
            )
        config = apply_overrides(
            config,
            parse_assignments(args.rate, "--rate"),
            seconds_overrides,
        )
        config = apply_remaining_units(
            config,
            parse_assignments(args.remaining_units, "--remaining-units"),
        )
        estimate = estimate_budget(config, skipped_workloads=args.skip)
        start_time = parse_start_time(args.start_time) if args.start_time else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(as_json(config, estimate, start_time))
    else:
        print_report(config, estimate, start_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
