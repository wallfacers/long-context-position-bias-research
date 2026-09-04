#!/usr/bin/env python3
"""Emit one read-only strict-queue progress, health, cost, and ETA snapshot."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


SEEDS = (20260825, 20260826, 20260827)
VARIANTS = (
    "independent_answer",
    "independent_evidence_id",
    "independent_evidence",
    "paired_answer",
    "paired_evidence_id",
    "paired_evidence",
)
TRAIN_FAMILIES = {
    "qwen_block96_training": "qwen_block96",
    "mistral_block96_training": "mistral_block96",
}
MULTISEED_RUNS = tuple(f"s{seed}_{variant}" for seed in SEEDS for variant in VARIANTS)
REPRESENTATIVE_RUNS = ("base", *VARIANTS)
LABELED_REPRESENTATIVE_RUNS = ("base", *(f"{variant}_block96" for variant in VARIANTS))
LABELED_MECHANISM_RUNS = (
    "base",
    "independent_answer_block96",
    "independent_evidence_block96",
    "paired_answer_block96",
    "paired_evidence_block96",
)
EVAL_WORKLOADS = {
    "qwen_block96_nolima": ("results/qwen_block96_nolima", MULTISEED_RUNS, 1050),
    "qwen_block96_longbench": ("results/qwen_block96_longbench", MULTISEED_RUNS, 600),
    "qwen_block96_mmlu": (
        "results/qwen_block96_mmlu",
        LABELED_REPRESENTATIVE_RUNS,
        14042,
    ),
    "qwen_block96_ifeval": ("results/qwen_block96_ifeval", REPRESENTATIVE_RUNS, 541),
    "qwen_block96_nolima_mechanisms": (
        "results/qwen_block96_nolima_mechanisms",
        LABELED_MECHANISM_RUNS,
        1350,
    ),
    "qwen_block96_rule": ("results/qwen_block96_rule", MULTISEED_RUNS, 4200),
    "mistral_block96_nolima": (
        "results/mistral_block96_nolima",
        ("base", *MULTISEED_RUNS),
        1050,
    ),
    "mistral_block96_longbench": (
        "results/mistral_block96_longbench",
        ("base", *MULTISEED_RUNS),
        600,
    ),
    "mistral_block96_mmlu": (
        "results/mistral_block96_mmlu",
        LABELED_REPRESENTATIVE_RUNS,
        14042,
    ),
    "mistral_block96_ifeval": (
        "results/mistral_block96_ifeval",
        REPRESENTATIVE_RUNS,
        541,
    ),
    "mistral_block96_nolima_mechanisms": (
        "results/mistral_block96_nolima_mechanisms",
        LABELED_MECHANISM_RUNS,
        1350,
    ),
    "mistral_block96_rule": (
        "results/mistral_block96_rule",
        ("base", *MULTISEED_RUNS),
        4200,
    ),
}
PROGRESS_PATTERN = re.compile(r"(?<!\d)(\d{1,4})/2000")
METRIC_PATTERN = re.compile(r"(\{[^\n\r]*'loss'[^\n\r]*\})")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(
            block.count(b"\n")
            for block in iter(lambda: handle.read(1024 * 1024), b"")
        )


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def completed_training_record(
    project_root: Path, family_root: str, seed: int, variant: str
) -> dict[str, Any] | None:
    output = project_root / "outputs" / family_root / f"seed_{seed}" / variant
    completion_path = output / "CANARY_COMPLETE.json"
    state_path = output / "checkpoint-96" / "trainer_state.json"
    if not completion_path.is_file() or not state_path.is_file():
        return None
    completion, state = read_json(completion_path), read_json(state_path)
    if (
        completion.get("schema_version") != "qlora-result-v1"
        or int(completion.get("global_step", -1)) != 96
        or int(state.get("global_step", -1)) != 96
    ):
        return None
    rows = [
        row
        for row in state.get("log_history", [])
        if "step" in row and "loss" in row
    ]
    steps = {int(row["step"]) for row in rows}
    if steps != set(range(1, 97)):
        return None
    loss = finite_values(rows, "loss")
    accuracy = finite_values(rows, "mean_token_accuracy")
    gradients = finite_values(rows, "grad_norm")
    all_finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("loss", "learning_rate", "grad_norm")
        if row.get(key) is not None
    )
    return {
        "family_root": family_root,
        "seed": seed,
        "variant": variant,
        "global_step": 96,
        "metric_rows": len(rows),
        "elapsed_seconds": float(completion.get("elapsed_seconds_this_invocation", 0)),
        "seconds_per_step": float(
            completion.get("seconds_per_step_this_invocation", 0)
        ),
        "first_loss": loss[0],
        "final_loss": loss[-1],
        "median_loss": statistics.median(loss),
        "median_loss_last_20": statistics.median(loss[-20:]),
        "final_token_accuracy": accuracy[-1] if accuracy else None,
        "median_token_accuracy": statistics.median(accuracy) if accuracy else None,
        "final_learning_rate": float(rows[-1]["learning_rate"]),
        "median_grad_norm_last_20": (
            statistics.median(gradients[-20:]) if gradients else None
        ),
        "all_metrics_finite": all_finite,
        "training_distribution_saturated": bool(
            statistics.median(loss) < 1e-4
            and accuracy
            and statistics.median(accuracy) > 0.999
        ),
    }


def process_table() -> str:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid,ppid,etime,stat,args", "--sort=pid"],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def active_training(project_root: Path, processes: str) -> dict[str, Any] | None:
    active_line = next(
        (line for line in processes.splitlines() if "scripts/train_qlora.py" in line),
        None,
    )
    if active_line is None:
        return None
    match = re.search(r"--output\s+(\S+)", active_line)
    if match is None:
        return {"process": active_line.strip(), "step": None}
    output = Path(match.group(1)).resolve()
    try:
        relative = output.relative_to(project_root)
    except ValueError:
        return {"process": active_line.strip(), "step": None}
    logs = sorted((output / "logs").glob("*.log"), key=lambda path: path.stat().st_mtime)
    text = logs[-1].read_text(encoding="utf-8", errors="replace") if logs else ""
    progress = [int(value) for value in PROGRESS_PATTERN.findall(text)]
    metric = None
    for candidate in METRIC_PATTERN.findall(text):
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            metric = parsed
    parts = relative.parts
    seed = int(parts[2].removeprefix("seed_")) if len(parts) >= 4 else None
    return {
        "output": relative.as_posix(),
        "family_root": parts[1] if len(parts) >= 2 else None,
        "seed": seed,
        "variant": parts[3] if len(parts) >= 4 else None,
        "step": min(96, max(progress, default=0)),
        "expected_step": 96,
        "latest_metric": metric,
        "process": active_line.strip(),
    }


def find_result(project_root: Path, relative_root: str, run_name: str) -> Path | None:
    root = project_root / relative_root
    direct = root / f"{run_name}.jsonl"
    if direct.is_file():
        return direct
    matches = sorted(root.glob(f"**/{run_name}.jsonl")) if root.is_dir() else []
    return matches[0] if len(matches) == 1 else None


def evaluation_progress(project_root: Path) -> dict[str, dict[str, Any]]:
    progress = {}
    for workload, (relative_root, run_names, rows_per_run) in EVAL_WORKLOADS.items():
        counts = {
            run_name: line_count(path)
            for run_name in run_names
            if (path := find_result(project_root, relative_root, run_name)) is not None
        }
        total = len(run_names) * rows_per_run
        completed = sum(min(rows_per_run, count) for count in counts.values())
        completion_path = project_root / relative_root / "completion.json"
        validated = False
        if completion_path.is_file():
            try:
                validated = read_json(completion_path).get("status") == "validated"
            except (OSError, json.JSONDecodeError):
                pass
        progress[workload] = {
            "result_root": relative_root,
            "runs_complete": sum(count == rows_per_run for count in counts.values()),
            "runs_expected": len(run_names),
            "rows_complete": completed,
            "rows_expected": total,
            "completion_fraction": completed / total,
            "validated_completion": validated,
            "partial_or_complete_counts": counts,
        }
    return progress


def gpu_probe() -> dict[str, Any] | None:
    query = "name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,pstate"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode:
        return None
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != 7:
        return None
    return dict(
        zip(
            ("name", "memory_used_mib", "memory_total_mib", "utilization_percent", "temperature_c", "power_draw_w", "pstate"),
            values,
            strict=True,
        )
    )


def scheduler_daemon_lines(processes: str) -> list[str]:
    pattern = re.compile(r"(?:^|\s)(?:/\S*/)?(?:cron|crond|atd)(?:\s|$)")
    return [line.strip() for line in processes.splitlines() if pattern.search(line)]


def runtime_safety_probe(status_root: Path, processes: str) -> dict[str, Any]:
    def command_lines(command: list[str]) -> int | None:
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=15,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if completed.returncode not in (0, 1):
            return None
        return len([line for line in completed.stdout.splitlines() if line.strip()])

    try:
        usage = shutil.disk_usage(status_root)
        disk = {
            "path": str(status_root.resolve()),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    except OSError:
        disk = None
    cron_lines = command_lines(["crontab", "-l"])
    at_lines = command_lines(["atq"])
    scheduler_daemons = scheduler_daemon_lines(processes)
    try:
        screen = subprocess.run(
            ["screen", "-ls"],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        screen_output = (screen.stdout + screen.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        screen_output = None
    return {
        "cron_nonempty_lines": cron_lines,
        "at_queue_lines": at_lines,
        "crontab_cli_available": shutil.which("crontab") is not None,
        "atq_cli_available": shutil.which("atq") is not None,
        "scheduler_daemon_processes": scheduler_daemons,
        "disk": disk,
        "screen_list": screen_output,
        "no_remote_scheduler_detected": (
            cron_lines in (None, 0)
            and at_lines in (None, 0)
            and not scheduler_daemons
        ),
    }


def estimate_remaining(
    budget: dict[str, Any],
    training: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    completed_records = training["completed_records"]
    measured_by_family = {
        family_root: [
            float(record["seconds_per_step"])
            for record in completed_records
            if record["family_root"] == family_root
            and float(record["seconds_per_step"]) > 0
        ]
        for family_root in TRAIN_FAMILIES.values()
    }
    active = training.get("active") or {}
    active_key = (active.get("family_root"), active.get("seed"), active.get("variant"))
    completed_keys = {
        (record["family_root"], record["seed"], record["variant"])
        for record in completed_records
    }
    conservative_remaining_hours = 0.0
    calibrated_remaining_hours = 0.0
    workload_rows = []
    for workload in budget["workloads"]:
        name = workload["name"]
        if name == "historical_qwen_longbench_residual_at_1458":
            fraction = 1.0
            conservative_hours = calibrated_hours = 0.0
        elif name in TRAIN_FAMILIES:
            family_root = TRAIN_FAMILIES[name]
            measured_steps = measured_by_family[family_root]
            measured_seconds_per_step = (
                statistics.median(measured_steps) if measured_steps else None
            )
            calibrated_seconds = 0.0
            remaining_equivalent_runs = 0.0
            for seed in SEEDS:
                for variant in VARIANTS:
                    key = (family_root, seed, variant)
                    if key in completed_keys:
                        continue
                    if key == active_key:
                        steps = int(active.get("step") or 0)
                        remaining_equivalent_runs += max(0, 96 - steps) / 96
                        if measured_seconds_per_step is not None:
                            calibrated_seconds += (
                                max(0, 96 - steps) * measured_seconds_per_step + 25
                            )
                    else:
                        remaining_equivalent_runs += 1
                        if measured_seconds_per_step is not None:
                            calibrated_seconds += 96 * measured_seconds_per_step + 25
            complete_steps = sum(
                96 for key in completed_keys if key[0] == family_root
            )
            if active_key and active_key[0] == family_root:
                complete_steps += int(active.get("step") or 0)
            fraction = min(1.0, complete_steps / (18 * 96))
            configured_per_run_hours = (
                int(workload["units_per_run"])
                * float(workload["seconds_per_unit"])
                / 3600
                + float(workload["fixed_hours_per_run"])
            )
            conservative_hours = remaining_equivalent_runs * configured_per_run_hours
            calibrated_hours = (
                calibrated_seconds / 3600
                if measured_seconds_per_step is not None
                else conservative_hours
            )
        elif name == "mistral_block96_materialization":
            complete = (
                training["project_root"]
                / "data/formal_block96_mistral7b_v03/completion.json"
            ).is_file()
            fraction = 1.0 if complete else 0.0
            conservative_hours = (
                0.0 if complete else float(workload["fixed_hours_per_run"])
            )
            calibrated_hours = conservative_hours
        elif name in evaluations:
            fraction = float(evaluations[name]["completion_fraction"])
            total_units = int(workload["runs"]) * int(workload["units_per_run"])
            conservative_hours = (
                total_units
                * (1.0 - fraction)
                * float(workload["seconds_per_unit"])
                / 3600
            )
            calibrated_hours = conservative_hours
        else:
            fraction = 0.0
            conservative_hours = int(workload["runs"]) * (
                int(workload["units_per_run"])
                * float(workload["seconds_per_unit"])
                / 3600
                + float(workload["fixed_hours_per_run"])
            )
            calibrated_hours = conservative_hours
        conservative_remaining_hours += conservative_hours
        calibrated_remaining_hours += calibrated_hours
        workload_rows.append(
            {
                "name": name,
                "completion_fraction": fraction,
                "conservative_remaining_gpu_hours": conservative_hours,
                "calibrated_remaining_gpu_hours": calibrated_hours,
            }
        )
    rate = float(next(iter(budget["pricing"].values()))["hourly_rate"])
    reserve = sum(float(item["amount"]) for item in budget.get("one_time_costs", []))
    expected_cost = calibrated_remaining_hours * rate + reserve
    conservative_cost = conservative_remaining_hours * rate + reserve
    contingency = float(budget["contingency_ratio"])
    return {
        "measured_training_seconds_per_step_by_family": {
            family_root: statistics.median(values) if values else None
            for family_root, values in measured_by_family.items()
        },
        "assumed_training_nonstep_overhead_seconds": 25,
        "remaining_gpu_hours": conservative_remaining_hours,
        "remaining_gpu_hours_measured_calibrated": calibrated_remaining_hours,
        "expected_finish": (
            now + timedelta(hours=calibrated_remaining_hours)
        ).isoformat(),
        "conservative_finish": (
            now + timedelta(hours=conservative_remaining_hours)
        ).isoformat(),
        "contingency_finish": (
            now
            + timedelta(hours=conservative_remaining_hours * (1 + contingency))
        ).isoformat(),
        "hourly_rate_cny": rate,
        "remaining_gpu_cost_plus_reserve_cny": expected_cost,
        "conservative_cost_plus_reserve_cny": conservative_cost,
        "remaining_cost_ceiling_cny": conservative_cost * (1 + contingency),
        "workloads": workload_rows,
    }


def parse_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now().astimezone()
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--now must include a UTC offset or Z")
    return parsed


def classify_queue_health(
    statuses: dict[str, str | None],
    process_chain_present: bool | None,
) -> str:
    top = (statuses.get("top") or "").lower()
    if top.startswith("failed"):
        return "failed"
    if top.startswith("validated"):
        return "validated"
    if "stage=qwen" in top:
        relevant_labels = ("top", "qwen", "qwen_training")
    elif "stage=mistral" in top:
        relevant_labels = ("top", "qwen", "mistral", "mistral_training")
    else:
        # Once both family queues have passed, top-level audit stages overwrite
        # the stage name. Historical subqueue status files are no longer live.
        relevant_labels = ("top",)
    relevant = [
        statuses[label].lower()
        for label in relevant_labels
        if statuses.get(label)
    ]
    if any(value.startswith("failed") for value in relevant):
        return "failed"
    if process_chain_present is True:
        return "running"
    if process_chain_present is False:
        return "stalled"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--status-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--budget-config", type=Path)
    parser.add_argument("--now", help="Timezone-aware timestamp for deterministic replay")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-system-probes", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise SystemExit("--project-root must be an existing directory")
    budget_path = (
        args.budget_config.resolve()
        if args.budget_config
        else project_root / "configs/autodl_strict_block96_budget.json"
    )
    try:
        budget = read_json(budget_path)
        now = parse_now(args.now)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    processes = "" if args.no_system_probes else process_table()
    active = None if args.no_system_probes else active_training(project_root, processes)
    completed_records = []
    by_family = {}
    for family_root in TRAIN_FAMILIES.values():
        family_records = []
        for seed in SEEDS:
            for variant in VARIANTS:
                record = completed_training_record(
                    project_root, family_root, seed, variant
                )
                if record is not None:
                    family_records.append(record)
                    completed_records.append(record)
        by_family[family_root] = {
            "conditions_complete": len(family_records),
            "conditions_expected": 18,
            "completion_fraction": len(family_records) / 18,
        }
    training = {
        "project_root": project_root,
        "conditions_complete": len(completed_records),
        "conditions_expected": 36,
        "by_family": by_family,
        "active": active,
        "completed_records": completed_records,
    }
    evaluations = evaluation_progress(project_root)
    status_names = {
        "top": "strict-block96-full-queue.status",
        "qwen": "qwen-block96-completion-queue.status",
        "mistral": "mistral-block96-completion-queue.status",
        "qwen_training": "qwen-block96-completion-artifacts/training.status",
        "mistral_training": "mistral-block96-completion-artifacts/training.status",
    }
    statuses = {}
    for label, relative in status_names.items():
        path = args.status_root / relative
        statuses[label] = path.read_text(errors="replace").strip() if path.is_file() else None
    eta = estimate_remaining(budget, training, evaluations, now)
    process_chain_present = (
        None
        if args.no_system_probes
        else "run_autodl_strict_block96_full_queue.sh" in processes
    )
    payload = {
        "schema_version": "strict-queue-progress-snapshot-v1",
        "status": "observed",
        "observed_at": now.isoformat(),
        "read_only_one_shot": True,
        "statuses": statuses,
        "queue_health": classify_queue_health(statuses, process_chain_present),
        "process_chain_present": process_chain_present,
        "training": training | {"project_root": str(project_root)},
        "evaluations": evaluations,
        "gpu": None if args.no_system_probes else gpu_probe(),
        "runtime_safety": (
            None
            if args.no_system_probes
            else runtime_safety_probe(args.status_root, processes)
        ),
        "estimate": eta,
        "assumptions": [
            "No daemon, polling loop, cron, at job, or power action is created.",
            "Evaluation ETA uses frozen per-suite throughput; completed JSONL rows are resumable units.",
            "Training ETA uses the median audited seconds/step plus 25 seconds per unfinished condition for load/save overhead.",
        ],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
