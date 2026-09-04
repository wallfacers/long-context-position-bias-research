#!/usr/bin/env python3
"""Requirement-by-requirement completion audit for the paper pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS = (
    "base",
    "paired_evidence",
    "paired_answer",
    "independent_evidence",
    "independent_answer",
)
TRAINED_RUNS = RUNS[1:]
POSITIONS = {"p000", "p010", "p025", "p050", "p075", "p090", "p100"}
CONTRASTS = {
    "paired_minus_independent_main_effect",
    "evidence_minus_answer_main_effect",
    "pairing_x_supervision_interaction",
}
DIRECT_POWER_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:sudo\s+)?(?:/usr/bin/)?(?:shutdown|poweroff|halt)(?:\s|$)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def record(checks: list[dict[str, Any]], name: str, evidence: Any) -> None:
    checks.append({"name": name, "status": "passed", "evidence": evidence})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(project_root: Path, results_dir: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    validation = read_json(results_dir / "validation-report.json")
    require(validation.get("status") == "validated", "full validation status is not validated")
    require(validation["matrix"]["total_samples"] == 21000, "validation total is not 21,000")
    require(validation["matrix"]["cells_per_run"] == 84, "validation does not have 84 cells")
    record(checks, "full_validation", validation["matrix"])

    reference_ids: set[str] | None = None
    reference_cells: Counter[tuple[Any, ...]] | None = None
    group_positions: dict[str, set[str]] = defaultdict(set)
    run_evidence: dict[str, Any] = {}
    for run_name in RUNS:
        path = results_dir / f"{run_name}.jsonl"
        rows = read_jsonl(path)
        require(len(rows) == 4200, f"{run_name} does not contain 4,200 rows")
        ids = {str(row["sample_id"]) for row in rows}
        require(len(ids) == 4200, f"{run_name} contains duplicate sample IDs")
        if reference_ids is None:
            reference_ids = ids
        else:
            require(ids == reference_ids, f"{run_name} sample IDs differ")
        cells = Counter(
            (
                row["task"],
                row["filler_type"],
                int(row["target_tokens"]),
                row["position_label"],
            )
            for row in rows
        )
        require(len(cells) == 84 and set(cells.values()) == {50}, f"{run_name} is not 84×50")
        if reference_cells is None:
            reference_cells = cells
        else:
            require(cells == reference_cells, f"{run_name} cells differ")
        if run_name == "base":
            for row in rows:
                group_positions[str(row["group_id"])].add(str(row["position_label"]))
        metadata = read_json(results_dir / f"{run_name}.jsonl.run.json")
        require(metadata.get("status") == "selection_complete", f"{run_name} metadata incomplete")
        require(metadata.get("max_new_tokens") == 176, f"{run_name} did not use output cap 176")
        require(metadata.get("seed") == 20260825, f"{run_name} seed mismatch")
        run_evidence[run_name] = {
            "rows": len(rows),
            "jsonl_sha256": sha256_file(path),
            "selection_sha256": metadata["selection_sha256"],
            "adapter_sha256": metadata.get("adapter_sha256"),
        }
    require(len(group_positions) == 600, "base does not contain 600 equivalence groups")
    require(all(value == POSITIONS for value in group_positions.values()), "groups lack seven positions")
    record(checks, "paired_test_matrix", run_evidence)

    training_summary = read_json(
        project_root / "outputs/training_diagnostics/training_metrics_summary.json"
    )
    require(training_summary.get("total_metric_rows") == 8000, "training metrics are not 8,000 steps")
    require(training_summary.get("nonfatal_errors") == [], "training metric export has errors")
    adapter_evidence = {}
    for run_name in TRAINED_RUNS:
        completion = read_json(project_root / f"outputs/{run_name}/TRAINING_COMPLETE.json")
        adapter = project_root / f"outputs/{run_name}/final_adapter/adapter_model.safetensors"
        require(completion.get("global_step") == 2000, f"{run_name} training is not at step 2,000")
        require(adapter.is_file() and adapter.stat().st_size > 0, f"{run_name} adapter missing")
        adapter_evidence[run_name] = {
            "global_step": completion["global_step"],
            "adapter_bytes": adapter.stat().st_size,
            "adapter_sha256": sha256_file(adapter),
        }
    record(checks, "training_and_adapters", adapter_evidence)

    analysis_path = results_dir / "analysis/ablation_analysis.json"
    analysis = read_json(analysis_path)
    require(analysis.get("scope", "").startswith("single-seed exploratory"), "scope caveat missing")
    require(analysis["bootstrap"]["replicates"] == 2000, "bootstrap is not 2,000 replicates")
    require(analysis["bootstrap"]["seed"] == 20260825, "bootstrap seed mismatch")
    require(CONTRASTS.issubset(analysis["contrasts"]), "2×2 contrasts missing")
    require(set(analysis["rows_per_run"].values()) == {4200}, "analysis row counts mismatch")
    screening = analysis["exploratory_screening"]
    require(screening["criteria"]["edge_accuracy_delta_floor"] == -0.02, "edge guardrail missing")
    require(screening["criteria"]["valid_json_floor"] == 0.99, "JSON guardrail missing")
    indices = results_dir / "analysis/paired_bootstrap_indices.jsonl.gz"
    require(indices.is_file() and indices.stat().st_size > 0, "bootstrap indices missing")
    record(
        checks,
        "paired_ablation_analysis",
        {
            "analysis_sha256": sha256_file(analysis_path),
            "bootstrap_indices_sha256": sha256_file(indices),
            "contrasts": sorted(CONTRASTS),
        },
    )

    figure_metadata = read_json(results_dir / "figures/figures.metadata.json")
    require(figure_metadata["analysis_sha256"] == sha256_file(analysis_path), "figure source hash mismatch")
    figure_files = []
    for stem in ("position_curves", "ablation_summary"):
        for suffix in ("png", "svg"):
            path = results_dir / "figures" / f"{stem}.{suffix}"
            require(path.is_file() and path.stat().st_size > 0, f"missing figure {path.name}")
            figure_files.append({"path": path.name, "sha256": sha256_file(path)})
    record(checks, "publication_figures", figure_files)

    reproducibility = results_dir / "reproducibility"
    required_repro = (
        "input-sha256.txt",
        "pip-freeze.txt",
        "nvidia-smi-q.txt",
        "uname.txt",
        "gpu-progress-telemetry.csv",
        "generation-cap-diagnostic.json",
        "cost-ledger.json",
    )
    repro_evidence = {}
    for name in required_repro:
        path = reproducibility / name
        require(path.is_file() and path.stat().st_size > 0, f"missing reproducibility/{name}")
        repro_evidence[name] = sha256_file(path)
    cap = read_json(reproducibility / "generation-cap-diagnostic.json")
    require(cap["lower_cap"]["max_new_tokens"] == 128, "lower cap diagnostic mismatch")
    require(cap["higher_cap"]["max_new_tokens"] == 176, "higher cap diagnostic mismatch")
    require(cap["matrix"] == {"cells": 84, "samples": 84, "samples_per_cell": 1}, "cap matrix mismatch")
    cost = read_json(reproducibility / "cost-ledger.json")
    require(len(cost["entries"]) >= 5, "cost ledger lacks task windows")
    telemetry_lines = sum(1 for _ in (reproducibility / "gpu-progress-telemetry.csv").open())
    require(telemetry_lines > 2, "GPU telemetry is empty")
    record(checks, "reproducibility_time_and_cost", repro_evidence | {"telemetry_rows": telemetry_lines - 1})

    report = results_dir / "paper-pilot-report.md"
    require(report.is_file() and report.stat().st_size > 0, "paper report missing")
    record(checks, "human_readable_report", {"sha256": sha256_file(report), "bytes": report.stat().st_size})

    offending = []
    for script in sorted((project_root / "scripts").glob("*.sh")):
        for number, line in enumerate(script.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("echo "):
                continue
            if DIRECT_POWER_COMMAND.search(line) or "shutdown_instance" in line:
                offending.append(f"{script.name}:{number}:{stripped}")
    require(not offending, f"script power commands found: {offending}")
    record(checks, "no_script_power_action", {"shell_scripts_scanned": len(list((project_root / 'scripts').glob('*.sh')))})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        checks = audit(args.project_root.resolve(), args.results_dir.resolve())
        status = "validated"
        error = None
    except Exception as exc:
        checks = []
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    payload = {
        "schema_version": "paper-pilot-completion-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "checks_passed": len(checks),
        "error": error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
