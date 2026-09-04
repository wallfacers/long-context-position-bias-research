#!/usr/bin/env python3
"""Fail closed unless every expected failure catalog matches its source rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any


EXPECTED_OUTPUTS = {
    "failure_case_catalog.csv",
    "failure_case_catalog.json",
    "failure_case_catalog.md",
}
FORBIDDEN_RAW_KEYS = {
    "prompt",
    "question",
    "target",
    "generated_text",
    "parsed",
    "answer",
    "evidence_quote",
    "sample_id",
    "group_id",
    "case_id",
    "book_id",
    "generated_text_sha256",
    "parsed_answer_sha256",
    "base_generated_text_sha256",
    "treatment_generated_text_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_inside(path: Path, root: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes project root: {path}") from exc
    return resolved


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid source JSON at {path}:{line_number}") from exc
            count += 1
    return count


def walk_keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def render_expected_views(catalog: dict[str, Any]) -> tuple[str, str]:
    """Rebuild the two human-readable views from the canonical safe JSON."""
    examples = catalog["examples"]
    counts = catalog["category_counts"]
    scopes = catalog["category_scopes"]
    denominators = catalog["scope_denominators"]
    rates = catalog["category_rates"]

    csv_buffer = io.StringIO()
    fieldnames = (
        "category",
        "scope",
        "run",
        "base_run",
        "treatment_run",
        "group_id_sha256",
        "sample_id_sha256",
    )
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for item in examples:
        rendered = {}
        for key in fieldnames:
            value = item.get(key, "")
            rendered[key] = (
                json.dumps(value, separators=(",", ":"))
                if isinstance(value, list)
                else value
            )
        writer.writerow(rendered)

    markdown = [
        "# License-safe failure case catalog",
        "",
        catalog["raw_text_policy"],
        "",
        "| Category | Scope | Candidates | Denominator | Rate | Selected fingerprints |",
        "|---|---|---:|---:|---:|---|",
    ]
    for category, count in sorted(counts.items()):
        identifiers = []
        for item in examples:
            if item["category"] != category:
                continue
            identifier = (
                item.get("group_id_sha256", "")
                if item["scope"] == "group"
                else item.get("sample_id_sha256", "")
            )
            if isinstance(identifier, list):
                identifier = identifier[0] if identifier else ""
            identifiers.append(str(identifier)[:12])
        scope = scopes[category]
        markdown.append(
            f"| `{category}` | `{scope}` | {count} | {denominators[scope]} | "
            f"{rates[category]:.6f} | {', '.join(identifiers)} |"
        )
    markdown.append("")
    return csv_buffer.getvalue(), "\n".join(markdown)


def audit_catalog(manifest_path: Path, project_root: Path) -> dict[str, Any]:
    manifest_path = require_inside(manifest_path, project_root, "manifest")
    if manifest_path.is_symlink():
        raise ValueError(f"Catalog manifest cannot be a symlink: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "failure-case-catalog-manifest-v1"
        or manifest.get("status") != "validated"
    ):
        raise ValueError(f"Invalid catalog manifest status/schema: {manifest_path}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != EXPECTED_OUTPUTS:
        raise ValueError(f"Catalog manifest output set differs: {manifest_path}")
    for name, record in outputs.items():
        path = require_inside(manifest_path.parent / name, project_root, name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked catalog output: {path}")
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"Catalog byte count mismatch: {path}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"Catalog output hash mismatch: {path}")

    catalog_path = manifest_path.parent / "failure_case_catalog.json"
    catalog_text = catalog_path.read_text(encoding="utf-8")
    catalog = json.loads(catalog_text)
    if (
        catalog.get("schema_version") != "license-safe-failure-case-catalog-v1"
        or catalog.get("status") != "validated"
    ):
        raise ValueError(f"Invalid catalog status/schema: {catalog_path}")
    leaked_keys = sorted(FORBIDDEN_RAW_KEYS.intersection(walk_keys(catalog)))
    if leaked_keys:
        raise ValueError(f"Catalog contains forbidden raw keys {leaked_keys}: {catalog_path}")
    counts = catalog.get("category_counts")
    scopes = catalog.get("category_scopes")
    denominators = catalog.get("scope_denominators")
    rates = catalog.get("category_rates")
    limit = catalog.get("max_examples_per_category")
    examples = catalog.get("examples")
    if (
        not isinstance(counts, dict)
        or not all(isinstance(value, int) and value >= 0 for value in counts.values())
        or not isinstance(scopes, dict)
        or set(scopes) != set(counts)
        or not set(scopes.values()).issubset({"row", "group", "cross_run"})
        or not isinstance(denominators, dict)
        or set(denominators) != {"row", "group", "cross_run"}
        or not all(
            isinstance(value, int) and value >= 0 for value in denominators.values()
        )
        or not isinstance(rates, dict)
        or set(rates) != set(counts)
        or not isinstance(limit, int)
        or limit <= 0
        or not isinstance(examples, list)
        or len(examples) > limit * len(counts)
    ):
        raise ValueError(f"Invalid category/selection accounting: {catalog_path}")
    for category, count in counts.items():
        denominator = denominators[scopes[category]]
        expected_rate = count / denominator if denominator else 0.0
        if not isinstance(rates[category], (int, float)) or not math.isclose(
            float(rates[category]), expected_rate, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"Category rate mismatch for {category}: {catalog_path}")
    selected_counts: dict[str, int] = {}
    for item in examples:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid selected example: {catalog_path}")
        category = item.get("category")
        scope = item.get("scope")
        if category not in counts or scope != scopes[category]:
            raise ValueError(f"Selected example category/scope mismatch: {catalog_path}")
        selected_counts[category] = selected_counts.get(category, 0) + 1
        if selected_counts[category] > limit:
            raise ValueError(f"Per-category selection limit exceeded: {catalog_path}")

    expected_csv, expected_markdown = render_expected_views(catalog)
    csv_path = manifest_path.parent / "failure_case_catalog.csv"
    markdown_path = manifest_path.parent / "failure_case_catalog.md"
    if csv_path.read_bytes() != expected_csv.encode("utf-8"):
        raise ValueError(f"Catalog CSV is not the canonical JSON-derived view: {csv_path}")
    if markdown_path.read_bytes() != expected_markdown.encode("utf-8"):
        raise ValueError(
            f"Catalog Markdown is not the canonical JSON-derived view: {markdown_path}"
        )

    sources = catalog.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Catalog has no sources: {catalog_path}")
    source_hashes: dict[str, str] = {}
    for source in sources:
        run = source.get("run")
        raw_path = source.get("path")
        if not isinstance(run, str) or not run or run in source_hashes:
            raise ValueError(f"Invalid or duplicate source run: {catalog_path}")
        if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
            raise ValueError(f"Source path must be relative: {catalog_path}")
        source_path = require_inside(project_root / raw_path, project_root, "source")
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(f"Missing or linked catalog source: {source_path}")
        actual_hash = sha256_file(source_path)
        if source.get("sha256") != actual_hash:
            raise ValueError(f"Catalog source hash mismatch: {source_path}")
        if source.get("rows") != count_jsonl_rows(source_path):
            raise ValueError(f"Catalog source row count mismatch: {source_path}")
        source_hashes[run] = actual_hash
    if manifest.get("source_sha256") != source_hashes:
        raise ValueError(f"Manifest/catalog source hash maps differ: {manifest_path}")

    return {
        "manifest": manifest_path.relative_to(project_root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "catalog_sha256": sha256_file(catalog_path),
        "source_runs": len(sources),
        "source_rows": sum(int(source["rows"]) for source in sources),
        "categories": len(counts),
        "selected_examples": len(examples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Explicit catalog manifest to audit. Repeat for a frozen primary set; "
            "this is mutually exclusive with recursive --results-root discovery."
        ),
    )
    parser.add_argument("--expected-catalogs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if not project_root.is_dir() or args.expected_catalogs <= 0:
        raise SystemExit("Project root and a positive expected count are required")
    if (args.results_root is None) == (not args.manifest):
        raise SystemExit(
            "Choose exactly one catalog selection mode: --results-root or repeated --manifest"
        )
    try:
        if args.manifest:
            manifests = sorted(
                {
                    require_inside(path, project_root, "manifest")
                    for path in args.manifest
                }
            )
            if len(manifests) != len(args.manifest):
                raise ValueError("Explicit failure-catalog manifest list contains duplicates")
        else:
            results_root = require_inside(args.results_root, project_root, "results root")
            if not results_root.is_dir():
                raise ValueError("Results root must be a directory")
            manifests = sorted(results_root.rglob("failure_case_catalog.manifest.json"))
        if len(manifests) != args.expected_catalogs:
            raise ValueError(
                f"Expected {args.expected_catalogs} failure catalogs, found {len(manifests)}"
            )
        records = [audit_catalog(path, project_root) for path in manifests]
        output = require_inside(args.output, project_root, "output")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    payload = {
        "schema_version": "failure-case-catalog-audit-v1",
        "status": "validated",
        "expected_catalogs": args.expected_catalogs,
        "catalogs": records,
        "total_source_rows": sum(record["source_rows"] for record in records),
        "total_selected_examples": sum(
            record["selected_examples"] for record in records
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Validated {len(records)} license-safe failure catalogs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
