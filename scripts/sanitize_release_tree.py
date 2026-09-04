#!/usr/bin/env python3
"""Copy an artifact tree into a portable release tree without machine-local paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_RUNTIME_ROOT = "/" + "root/autodl-tmp"
LOCAL_HOME_ROOT = "/" + "home/"
TEXT_SUFFIXES = {
    ".bbl",
    ".bib",
    ".cfg",
    ".cls",
    ".csv",
    ".jinja",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".sha256",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = (
    re.compile(r"(?:password|密码)\s*[:：=]", re.IGNORECASE),
    re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
)
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])/(?:root|home)/[^\s\"'{}]+")
REWRITES = (
    (re.compile(re.escape(str(ROOT.resolve())) + r"/"), ""),
    (re.compile(re.escape(REMOTE_RUNTIME_ROOT + "/position-bias-pilot/")), ""),
    (
        re.compile(
            re.escape(LOCAL_HOME_ROOT)
            + r"[^/]+/(?:[^/]+/)*long-context-position-bias-research/"
        ),
        "",
    ),
    (re.compile(re.escape(REMOTE_RUNTIME_ROOT + "/models/")), "models/"),
    (
        re.compile(re.escape(REMOTE_RUNTIME_ROOT) + r"(?=/|\s|[\"'{}]|$)"),
        "runtime",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_text(text: str) -> tuple[str, int]:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError("Potential credential material found; refusing to redact silently")
    total = 0
    for pattern, replacement in REWRITES:
        text, count = pattern.subn(replacement, text)
        total += count
    remaining = ABSOLUTE_PATH.search(text)
    if remaining:
        raise ValueError(f"Unrecognized machine-local path remains: {remaining.group(0)}")
    return text, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_dir():
        raise SystemExit("--input must be an existing artifact directory")
    if output.exists():
        raise SystemExit("--output already exists; release sanitization never overwrites")
    if source == output or source in output.parents:
        raise SystemExit("--output cannot be inside --input")
    output.mkdir(parents=True)
    records = []
    try:
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symlink is forbidden: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            replacements = 0
            if path.suffix.lower() in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8")
                text, replacements = sanitize_text(text)
                destination.write_text(text, encoding="utf-8")
            else:
                shutil.copyfile(path, destination)
            records.append(
                {
                    "path": relative.as_posix(),
                    "bytes": destination.stat().st_size,
                    "source_sha256": sha256_file(path),
                    "release_sha256": sha256_file(destination),
                    "path_replacements": replacements,
                }
            )
    except Exception:
        # The output was newly created by this process and is not an input/user tree.
        shutil.rmtree(output)
        raise
    manifest = {
        "schema_version": "sanitized-release-tree-v1",
        "status": "validated",
        "source_tree_name": source.name,
        "files": records,
        "files_with_replacements": sum(item["path_replacements"] > 0 for item in records),
        "total_path_replacements": sum(item["path_replacements"] for item in records),
        "secret_policy": "fail closed; credentials are never silently redacted",
    }
    manifest_path = output / "sanitization_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(f"Sanitized {len(records)} files with {manifest['total_path_replacements']} path rewrites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
