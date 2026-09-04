#!/usr/bin/env python3
"""Freeze the official IFEval prompts and verifier metadata for regression testing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


OFFICIAL_REVISION = "041338718b4e8151372fd63677104c65b73a0a4e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.source.open(encoding="utf-8") if line.strip()
    ]
    if len(rows) != 541:
        raise SystemExit(f"Expected 541 official IFEval prompts, found {len(rows)}")
    keys = [int(row["key"]) for row in rows]
    prompts = [str(row["prompt"]) for row in rows]
    if len(keys) != len(set(keys)) or len(prompts) != len(set(prompts)):
        raise SystemExit("IFEval keys and prompts must be unique")
    output_rows = []
    families = Counter()
    for row in rows:
        for instruction_id in row["instruction_id_list"]:
            families[str(instruction_id).split(":", 1)[0]] += 1
        output_rows.append(
            {
                "schema_version": "ifeval-prompt-v1",
                "sample_id": f"ifeval/{int(row['key'])}",
                "key": int(row["key"]),
                "prompt": str(row["prompt"]),
                "instruction_id_list": [str(value) for value in row["instruction_id_list"]],
                "kwargs": row["kwargs"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "ifeval-regression-manifest-v1",
        "status": "validated",
        "official_repository": "https://github.com/google-research/google-research",
        "official_revision": OFFICIAL_REVISION,
        "official_source": str(args.source),
        "official_source_sha256": sha256_file(args.source),
        "license": "Apache-2.0",
        "rows": len(output_rows),
        "instruction_instances": sum(len(row["instruction_id_list"]) for row in output_rows),
        "instruction_family_counts": dict(sorted(families.items())),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "scoring": "official strict and loose prompt-level and instruction-level IFEval verifiers",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote IFEval regression set: rows={len(output_rows)} "
        f"instructions={manifest['instruction_instances']} sha256={manifest['output_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
