#!/usr/bin/env python3
"""Atomically rebuild failure-catalog views from their canonical safe JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audit_failure_case_catalogs import (
    EXPECTED_OUTPUTS,
    FORBIDDEN_RAW_KEYS,
    render_expected_views,
    sha256_file,
    walk_keys,
)


def require_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.catalog_dir.resolve()
    if not directory.is_dir() or args.catalog_dir.is_symlink():
        raise SystemExit(f"Catalog directory must be a real directory: {directory}")

    json_path = directory / "failure_case_catalog.json"
    csv_path = directory / "failure_case_catalog.csv"
    markdown_path = directory / "failure_case_catalog.md"
    manifest_path = directory / "failure_case_catalog.manifest.json"
    try:
        for path, label in (
            (json_path, "catalog JSON"),
            (csv_path, "catalog CSV"),
            (markdown_path, "catalog Markdown"),
            (manifest_path, "catalog manifest"),
        ):
            require_regular(path, label)
        catalog = json.loads(json_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            catalog.get("schema_version")
            != "license-safe-failure-case-catalog-v1"
            or catalog.get("status") != "validated"
        ):
            raise ValueError("Catalog JSON is not validated")
        leaked_keys = sorted(FORBIDDEN_RAW_KEYS.intersection(walk_keys(catalog)))
        if leaked_keys:
            raise ValueError(f"Catalog contains forbidden raw keys: {leaked_keys}")
        if (
            manifest.get("schema_version")
            != "failure-case-catalog-manifest-v1"
            or manifest.get("status") != "validated"
            or not isinstance(manifest.get("outputs"), dict)
            or set(manifest["outputs"]) != EXPECTED_OUTPUTS
        ):
            raise ValueError("Catalog manifest is not validated")
        expected_csv, expected_markdown = render_expected_views(catalog)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    # The manifest is replaced last. An interruption can therefore only produce a
    # hash mismatch, which the aggregate audit rejects rather than silently accepts.
    atomic_write(csv_path, expected_csv.encode("utf-8"))
    atomic_write(markdown_path, expected_markdown.encode("utf-8"))
    manifest["outputs"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (json_path, csv_path, markdown_path)
    }
    atomic_write(
        manifest_path,
        (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    print(f"Re-rendered canonical failure-catalog views in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
