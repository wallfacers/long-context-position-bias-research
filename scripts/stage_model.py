#!/usr/bin/env python3
"""Download a pinned Hugging Face model and build an upload integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "qwen25_7b_model.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_manifest(output: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name
        not in {
            "model_manifest.json",
            "model_integrity_attestation.json",
            "chat_protocol_audit.json",
        }
        and not {".cache", ".git"}.intersection(path.relative_to(output).parts)
    ]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Rebuild hashes for an already complete local model without downloading.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        snapshot_download(
            repo_id=config["model_id"],
            revision=config["revision"],
            local_dir=str(args.output),
        )
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (args.output / name).is_file()]
    has_weights = bool(
        list(args.output.glob("*.safetensors"))
        or (args.output / "model.safetensors.index.json").is_file()
    )
    if missing or not has_weights:
        raise SystemExit(
            "Incomplete model directory: "
            + ", ".join(missing + ([] if has_weights else ["model weights"]))
        )
    actual_model_config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": expected, "actual": actual_model_config.get(key)}
        for key, expected in config["config_signature"].items()
        if actual_model_config.get(key) != expected
    }
    if mismatches:
        raise SystemExit("Model config signature mismatch: " + json.dumps(mismatches))
    files = artifact_manifest(args.output)
    manifest = {
        "schema_version": "local-model-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config["model_id"],
        "revision": config["revision"],
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    manifest_path = args.output / "model_manifest.json"
    write_json_atomic(manifest_path, manifest)
    attestation = {
        "schema_version": "model-integrity-attestation-v1",
        "model": str(args.output.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "revision": config["revision"],
        "file_state": [
            {
                "path": item["path"],
                "bytes": (args.output / item["path"]).stat().st_size,
                "mtime_ns": (args.output / item["path"]).stat().st_mtime_ns,
            }
            for item in files
        ],
        "provenance": (
            "created immediately after stage_model full SHA-256 manifest generation; "
            "safe to reuse only while manifest, sizes, and mtimes are unchanged"
        ),
    }
    write_json_atomic(args.output / "model_integrity_attestation.json", attestation)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
