#!/usr/bin/env python3
"""Fail-fast AutoDL checks before loading a model or beginning paid work."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from position_bias_research.chat_protocol import NATIVE, selected_protocol_for_tokenizer


REQUIRED = {
    "train": ("torch", "transformers", "datasets", "trl", "peft", "bitsandbytes", "accelerate"),
    "eval": ("torch", "transformers", "vllm"),
}


def tokenizer_fingerprint(tokenizer: Any) -> str:
    payload = {
        "backend": tokenizer.backend_tokenizer.to_str(),
        "chat_template": tokenizer.chat_template,
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def verify_model_contract(model: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Bind the runtime snapshot to the model contract stored with the data."""
    actual_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    signature = contract.get("config_signature", {})
    mismatches = {
        key: {"expected": expected, "actual": actual_config.get(key)}
        for key, expected in signature.items()
        if actual_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Runtime model config violates the data contract: " + json.dumps(mismatches)
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    fingerprint = tokenizer_fingerprint(tokenizer)
    expected_fingerprint = contract.get("tokenizer_fingerprint")
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError(
            "Runtime tokenizer fingerprint differs from the tokenizer pinned with the data"
        )
    protocol = selected_protocol_for_tokenizer(tokenizer)
    expected_protocol = contract.get("chat_protocol")
    if expected_protocol and protocol != expected_protocol:
        raise ValueError(
            f"Runtime chat protocol {protocol} != pinned protocol {expected_protocol}"
        )
    audit_path = model / "chat_protocol_audit.json"
    expected_audit = contract.get("chat_protocol_audit_sha256")
    if expected_audit:
        if not audit_path.is_file():
            raise ValueError("Pinned model contract requires chat_protocol_audit.json")
        actual_audit = sha256_file(audit_path)
        if actual_audit != expected_audit:
            raise ValueError(
                "Runtime chat-protocol audit differs from the audit pinned with the data"
            )
    return {
        "tokenizer_fingerprint": fingerprint,
        "chat_protocol": protocol,
        "chat_protocol_audit_sha256": (
            sha256_file(audit_path) if audit_path.is_file() else None
        ),
        "config_signature_verified": bool(signature),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_chat_protocol(metadata: dict[str, Any]) -> str:
    """Interpret legacy missing/null protocol metadata as the native protocol."""
    return metadata.get("chat_protocol") or NATIVE


def existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ValueError(f"No existing parent for {path}")
        candidate = candidate.parent
    return candidate


def model_stat_snapshot(model: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = []
    for artifact in manifest.get("files", []):
        path = model / artifact["path"]
        stat = path.stat()
        snapshot.append(
            {
                "path": artifact["path"],
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return snapshot


def verify_model(
    model: Path, attestation_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (model / name).is_file()]
    has_weights = bool(
        list(model.glob("*.safetensors"))
        or list(model.glob("*.bin"))
        or (model / "model.safetensors.index.json").is_file()
        or (model / "pytorch_model.bin.index.json").is_file()
    )
    if missing or not has_weights:
        details = ", ".join(missing + ([] if has_weights else ["model weights"]))
        raise ValueError(f"Incomplete local model directory {model}: missing {details}")
    manifest_path = model / "model_manifest.json"
    manifest_verified = False
    verification_mode = "required-files-only"
    revision = None
    attestation: dict[str, Any] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_sha256 = sha256_file(manifest_path)
        if attestation_path is not None and attestation_path.is_file():
            cached = json.loads(attestation_path.read_text(encoding="utf-8"))
            if cached.get("schema_version") != "model-integrity-attestation-v1":
                raise ValueError(f"Unsupported model attestation: {attestation_path}")
            if Path(cached.get("model", "")).resolve() != model.resolve():
                raise ValueError("Model attestation points to a different model directory")
            if cached.get("manifest_sha256") != manifest_sha256:
                raise ValueError("Model manifest changed after the integrity attestation")
            expected_state = cached.get("file_state", [])
            actual_state = model_stat_snapshot(model, manifest)
            if actual_state != expected_state:
                raise ValueError("Model files changed after the integrity attestation")
            verification_mode = "cached-sha256-with-stat-revalidation"
            attestation = cached
        else:
            for artifact in manifest.get("files", []):
                artifact_path = model / artifact["path"]
                if not artifact_path.is_file():
                    raise ValueError(f"Missing model artifact: {artifact_path}")
                if artifact_path.stat().st_size != artifact["bytes"]:
                    raise ValueError(f"Size mismatch for model artifact: {artifact_path}")
                if sha256_file(artifact_path) != artifact["sha256"]:
                    raise ValueError(f"SHA-256 mismatch for model artifact: {artifact_path}")
            verification_mode = "full-sha256"
            attestation = {
                "schema_version": "model-integrity-attestation-v1",
                "model": str(model.resolve()),
                "manifest_sha256": manifest_sha256,
                "revision": manifest.get("revision"),
                "file_state": model_stat_snapshot(model, manifest),
            }
        manifest_verified = True
        revision = manifest.get("revision")
    return (
        {
            "path": str(model.resolve()),
            "files": len(list(model.iterdir())),
            "revision": revision,
            "manifest_verified": manifest_verified,
            "verification_mode": verification_mode,
        },
        attestation,
    )


def manifest_entry(manifest_path: Path, data_path: Path) -> dict[str, Any] | None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_root = manifest_path.parent.resolve()
    resolved_data = data_path.resolve()
    for entry in manifest.get("files", []):
        if (manifest_root / entry["path"]).resolve() == resolved_data:
            return entry
    return None


def verify_data(data: Path, manifest: Path | None, model: Path) -> dict[str, Any]:
    if data.is_file():
        report: dict[str, Any] = {
            "path": str(data.resolve()),
            "bytes": data.stat().st_size,
        }
        if manifest:
            entry = manifest_entry(manifest, data)
            if entry is None:
                raise ValueError(f"{data} is not listed in {manifest}")
            actual = sha256_file(data)
            if actual != entry["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {data}: {actual} != {entry['sha256']}")
            report["sha256"] = actual
            report["manifest_verified"] = True
        return report

    metadata_path = data / "pretokenization.json"
    state_path = data / "state.json"
    if not metadata_path.is_file() or not state_path.is_file():
        raise ValueError(f"Not a pre-tokenized dataset directory: {data}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for artifact in metadata.get("artifact_files", []):
        artifact_path = data / artifact["path"]
        if not artifact_path.is_file():
            raise ValueError(f"Missing pre-tokenized artifact: {artifact_path}")
        if artifact_path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Size mismatch for pre-tokenized artifact: {artifact_path}")
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"SHA-256 mismatch for pre-tokenized artifact: {artifact_path}")
    from transformers import AutoTokenizer

    local_tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    actual_tokenizer_fingerprint = tokenizer_fingerprint(local_tokenizer)
    if metadata.get("tokenizer_fingerprint") != actual_tokenizer_fingerprint:
        raise ValueError(
            "Pre-tokenized data does not match the tokenizer/chat template in the local model"
        )
    actual_chat_protocol = selected_protocol_for_tokenizer(local_tokenizer)
    if metadata_chat_protocol(metadata) != actual_chat_protocol:
        raise ValueError(
            "Pre-tokenized data does not match the selected model chat compatibility protocol"
        )
    if manifest:
        source_name = Path(metadata["source"]).name
        entries = [entry for entry in json.loads(manifest.read_text(encoding="utf-8"))["files"] if Path(entry["path"]).name == source_name]
        matches = [entry for entry in entries if entry["sha256"] == metadata["source_sha256"]]
        if len(matches) != 1:
            raise ValueError(
                f"Pre-tokenized source hash for {data} does not uniquely match {manifest}"
            )
    return {
        "path": str(data.resolve()),
        "rows": metadata["rows"],
        "source_sha256": metadata["source_sha256"],
        "chat_protocol": actual_chat_protocol,
        "manifest_verified": bool(manifest),
    }


def package_versions(mode: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for package in REQUIRED[mode]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise ValueError("Missing packages: " + ", ".join(missing))
    return versions


def gpu_report(min_vram_gb: float) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    count = torch.cuda.device_count()
    properties = torch.cuda.get_device_properties(0)
    vram_gb = properties.total_memory / 1024**3
    if vram_gb < min_vram_gb:
        raise ValueError(
            f"GPU 0 has {vram_gb:.1f} GiB VRAM, below required {min_vram_gb:.1f} GiB"
        )
    if not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not report bfloat16 support")
    return {
        "cuda_version": torch.version.cuda,
        "device_count": count,
        "gpu_0": properties.name,
        "gpu_0_vram_gib": round(vram_gb, 2),
        "bf16_supported": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(REQUIRED), required=True)
    parser.add_argument("--model", type=Path, required=True, help="Existing local model directory")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-vram-gb", type=float)
    parser.add_argument("--min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--require-model-manifest", action="store_true")
    parser.add_argument(
        "--model-attestation",
        type=Path,
        help="Reuse a prior full-hash attestation after revalidating manifest, sizes, and mtimes.",
    )
    parser.add_argument(
        "--write-model-attestation",
        type=Path,
        help="Write the model integrity attestation created by a full SHA-256 verification.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"Model must be an existing local directory; remote IDs are forbidden: {args.model}")
    if not args.data.exists():
        raise SystemExit(f"Missing dataset: {args.data}")
    if args.manifest and not args.manifest.is_file():
        raise SystemExit(f"Missing manifest: {args.manifest}")

    disk = shutil.disk_usage(existing_parent(args.output))
    free_gib = disk.free / 1024**3
    if free_gib < args.min_free_disk_gb:
        raise SystemExit(
            f"Only {free_gib:.1f} GiB free for {args.output}; require {args.min_free_disk_gb:.1f} GiB"
        )
    minimum_vram = args.min_vram_gb
    if minimum_vram is None:
        minimum_vram = 30.0 if args.mode == "train" else 22.0
    try:
        model_report, model_attestation = verify_model(args.model, args.model_attestation)
        if args.require_model_manifest and not model_report["manifest_verified"]:
            raise ValueError(
                "Local model lacks model_manifest.json; run scripts/stage_model.py --manifest-only"
            )
        model_contract_report = None
        if args.manifest:
            data_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            model_contract = data_manifest.get("model", {})
            expected_revision = model_contract.get("revision")
            if (
                expected_revision
                and model_report["revision"]
                and model_report["revision"] != expected_revision
            ):
                raise ValueError(
                    f"Model revision {model_report['revision']} != dataset revision {expected_revision}"
                )
            if model_contract:
                model_contract_report = verify_model_contract(args.model, model_contract)
        report = {
            "status": "ready",
            "mode": args.mode,
            "model": model_report,
            "model_contract": model_contract_report,
            "data": verify_data(args.data, args.manifest, args.model),
            "packages": package_versions(args.mode),
            "gpu": gpu_report(minimum_vram),
            "output_parent_free_gib": round(free_gib, 2),
        }
    except ValueError as exc:
        raise SystemExit(f"PREFLIGHT FAILED: {exc}") from exc
    if args.write_model_attestation:
        if model_attestation is None:
            raise SystemExit("PREFLIGHT FAILED: no model manifest is available for attestation")
        args.write_model_attestation.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.write_model_attestation.with_name(
            args.write_model_attestation.name + ".tmp"
        )
        temporary.write_text(
            json.dumps(model_attestation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.write_model_attestation)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
