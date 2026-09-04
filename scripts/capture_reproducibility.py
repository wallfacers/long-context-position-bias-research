#!/usr/bin/env python3
"""Capture a secret-safe, machine-readable reproducibility record for a run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "HF_HUB_OFFLINE",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_OFFLINE",
)
PACKAGE_NAMES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface-hub",
    "matplotlib",
    "numpy",
    "peft",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
    "trl",
)
VARIANTS = (
    "paired_evidence",
    "paired_evidence_id",
    "paired_answer",
    "independent_evidence",
    "independent_evidence_id",
    "independent_answer",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root) if root else path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": command, "available": False}
    result = subprocess.run(
        [executable, *command[1:]],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return {
        "command": command,
        "available": True,
        "returncode": result.returncode,
        "output": result.stdout.strip(),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def code_manifest(project_root: Path) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    source_suffixes = {".json", ".md", ".py", ".sh", ".toml", ".txt"}
    for directory in ("scripts", "src", "configs", "docs", "tests"):
        root = project_root / directory
        if root.is_dir():
            candidates.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in source_suffixes
            )
    for name in (
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "requirements-train.txt",
        "requirements-eval.txt",
    ):
        path = project_root / name
        if path.is_file():
            candidates.append(path)
    return [file_record(path, project_root) for path in sorted(set(candidates))]


def output_records(output_root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for variant in VARIANTS:
        root = output_root / variant
        entries: dict[str, Any] = {}
        for relative in (
            "run_config.json",
            "TRAINING_COMPLETE.json",
            "final_adapter/adapter_config.json",
            "final_adapter/adapter_model.safetensors",
            "final_adapter/tokenizer_config.json",
        ):
            path = root / relative
            if path.is_file():
                entries[relative] = file_record(path, root)
        invocation_dir = root / "invocations"
        if invocation_dir.is_dir():
            entries["invocations"] = [
                file_record(path, root)
                for path in sorted(invocation_dir.glob("*.json"))
                if path.is_file()
            ]
        records[variant] = entries
    return records


def model_identity(model: Path) -> dict[str, Any]:
    manifest_path = model / "model_manifest.json"
    config_path = model / "config.json"
    record: dict[str, Any] = {"path": str(model.resolve())}
    if manifest_path.is_file():
        record["manifest_file"] = file_record(manifest_path)
        record["manifest"] = read_json(manifest_path)
    if config_path.is_file():
        record["config_file"] = file_record(config_path)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--loss-health", type=Path)
    parser.add_argument("--queue-log", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    required = (args.project_root, args.model, args.output_root)
    if not all(path.is_dir() for path in required):
        raise SystemExit("--project-root, --model, and --output-root must be directories")
    if not args.data_manifest.is_file():
        raise SystemExit(f"Missing data manifest: {args.data_manifest}")

    try:
        import torch

        torch_record: dict[str, Any] = {
            "version": torch.__version__,
            "built_with_cuda": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        }
    except ImportError:
        torch_record = {"available": False}

    copied_runtime_files: dict[str, Any] = {}
    runtime_copies = (
        (args.telemetry, args.output.parent / "hardware_telemetry.csv"),
        (args.loss_health, args.output.parent / "training_loss_health.jsonl"),
        (args.queue_log, args.output.parent / "train_queue.log"),
    )
    for source, destination in runtime_copies:
        if source is None:
            continue
        if not source.is_file():
            raise SystemExit(f"Missing requested runtime record: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        copied_runtime_files[destination.name] = file_record(destination)

    data_manifest = read_json(args.data_manifest)
    record = {
        "schema_version": "training-reproducibility-v1",
        "captured_at": utc_now(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "uname": list(platform.uname()),
        },
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit,pci.device_id,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "torch": torch_record,
        "packages": package_versions(),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
        "safe_environment": {key: os.environ.get(key) for key in SAFE_ENVIRONMENT_KEYS},
        "model": model_identity(args.model),
        "data_manifest_file": file_record(args.data_manifest),
        "data_manifest": data_manifest,
        "code_manifest": code_manifest(args.project_root),
        "training_outputs": output_records(args.output_root),
        "runtime_records": copied_runtime_files,
        "reproducibility_scope": {
            "training_seed": 20260825,
            "seed_source": "each output variant's run_config.json is authoritative",
            "determinism_note": (
                "Seeds and exact artifacts are recorded. CUDA, quantized kernels, and parallel "
                "floating-point reductions may still prevent bit-for-bit equality; reproduce "
                "claims from saved per-sample predictions and confidence intervals."
            ),
            "training_metrics_note": (
                "Training loss is completion-only. It is diagnostic and must not substitute "
                "for held-out position-equivalent evaluation."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Wrote reproducibility record to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
