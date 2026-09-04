#!/usr/bin/env python3
"""Create a checksum manifest for prepared data before cloud upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "qwen25_7b_model.json"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    files = sorted(
        path
        for path in root.rglob("*.jsonl")
        if path.is_file() and path.resolve() != output
    )
    if not files:
        raise SystemExit(f"No JSONL files found under {root}")
    entries = []
    for path in files:
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "lines": line_count(path),
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": "data-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "model": model_config,
        "files": entries,
        "totals": {
            "files": len(entries),
            "bytes": sum(entry["bytes"] for entry in entries),
            "lines": sum(entry["lines"] for entry in entries),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(f"Wrote manifest for {len(entries)} files to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
