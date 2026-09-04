#!/usr/bin/env python3
"""Build a deterministic, license-safe catalog of qualitative failure patterns.

The catalog deliberately excludes prompts, targets, generated text, parsed text,
benchmark answers, raw identifiers, and absolute paths. It retains structural
metadata, scored flags, and one-way SHA-256 fingerprints needed to trace a
selected case back to the private/audited prediction files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SAFE_FIELDS = (
    "task",
    "evaluation_mode",
    "filler_type",
    "target_tokens",
    "position_label",
    "target_position",
    "actual_position",
    "finish_reason",
)
TEXT_KEYS_FORBIDDEN = {
    "prompt",
    "question",
    "target",
    "generated_text",
    "parsed",
    "answer",
    "evidence_quote",
}
SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.:/+\-]{1,128}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str | None:
    if value is None:
        return None
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def portable_source_path(path: Path) -> str:
    """Return a non-absolute source label without leaking host layout."""
    root = Path.cwd().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--run expects NAME=JSONL, got: {value}")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError(f"--run expects non-empty NAME=JSONL, got: {value}")
    if not SAFE_LABEL_RE.fullmatch(name.strip()):
        raise ValueError(f"--run NAME must be a portable identifier, got: {name}")
    return name.strip(), Path(raw_path).resolve()


def load_run(name: str, path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing result JSONL: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            for required in ("sample_id", "group_id", "answer_correct", "valid_json"):
                if required not in row:
                    raise ValueError(f"{path}:{line_number} lacks {required}")
            for required_bool in ("answer_correct", "valid_json"):
                if not isinstance(row[required_bool], bool):
                    raise ValueError(
                        f"{path}:{line_number} {required_bool} must be boolean"
                    )
            for optional_bool in (
                "evidence_quotes_correct",
                "all_predicted_quotes_supported",
                "evidence_quotes_applicable",
                "all_predicted_quotes_supported_applicable",
            ):
                if row.get(optional_bool) is not None and not isinstance(
                    row[optional_bool], bool
                ):
                    raise ValueError(
                        f"{path}:{line_number} {optional_bool} must be boolean or null"
                    )
            sample_id = str(row["sample_id"])
            group_id = str(row["group_id"])
            if not sample_id or not group_id:
                raise ValueError(f"{path}:{line_number} has an empty identifier")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id in {name}: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty result JSONL: {path}")
    return rows


def safe_context(row: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in SAFE_FIELDS:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float, bool)) or (
            isinstance(value, str) and SAFE_LABEL_RE.fullmatch(value)
        ):
            context[key] = value
        else:
            context[f"{key}_sha256"] = fingerprint(value)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("length_label", "length_tokens"):
            value = metadata.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float, bool)) or (
                isinstance(value, str) and SAFE_LABEL_RE.fullmatch(value)
            ):
                context[key] = value
            else:
                context[f"{key}_sha256"] = fingerprint(value)
        for key in ("case_id", "book_id"):
            if metadata.get(key) is not None:
                context[f"{key}_sha256"] = fingerprint(metadata[key])
    return context


def scored_flags(row: dict[str, Any]) -> dict[str, bool | None]:
    quote_applicable = bool(row.get("evidence_quotes_applicable", True))
    support_applicable = bool(
        row.get("all_predicted_quotes_supported_applicable", quote_applicable)
    )
    return {
        "valid_json": bool(row["valid_json"]),
        "answer_correct": bool(row["answer_correct"]),
        "evidence_quotes_correct": (
            None
            if not quote_applicable or row.get("evidence_quotes_correct") is None
            else bool(row["evidence_quotes_correct"])
        ),
        "all_predicted_quotes_supported": (
            None
            if not support_applicable
            or row.get("all_predicted_quotes_supported") is None
            else bool(row["all_predicted_quotes_supported"])
        ),
    }


def row_categories(row: dict[str, Any]) -> list[str]:
    flags = scored_flags(row)
    categories: list[str] = []
    if not flags["valid_json"]:
        categories.append("invalid_json")
    if not flags["answer_correct"]:
        categories.append("answer_wrong")
    quote = flags["evidence_quotes_correct"]
    if flags["answer_correct"] and quote is False:
        categories.append("answer_correct_quote_wrong")
    if not flags["answer_correct"] and quote is True:
        categories.append("answer_wrong_quote_correct")
    if row.get("finish_reason") == "length":
        categories.append("generation_hit_length_cap")
    return categories


def position_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    for key in ("actual_position", "target_position"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value), str(row["sample_id"])
    return float("inf"), str(row["sample_id"])


def row_example(run: str, category: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": "row",
        "category": category,
        "run": run,
        "sample_id_sha256": fingerprint(str(row["sample_id"])),
        "group_id_sha256": fingerprint(str(row["group_id"])),
        "context": safe_context(row),
        "flags": scored_flags(row),
    }


def group_examples(run: str, rows: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    for group_id, raw_group in sorted(groups.items()):
        group = sorted(raw_group, key=position_sort_key)
        if len(group) < 3:
            continue
        edge = (group[0], group[-1])
        middle = group[1:-1]
        categories: list[str] = []
        if any(bool(row["answer_correct"]) for row in edge) and any(
            not bool(row["answer_correct"]) for row in middle
        ):
            categories.append("edge_success_middle_failure")
        if any(bool(row["answer_correct"]) for row in middle) and any(
            not bool(row["answer_correct"]) for row in edge
        ):
            categories.append("middle_success_edge_failure")
        answer_fingerprints = {
            fingerprint(row.get("parsed", {}).get("answer"))
            for row in group
            if isinstance(row.get("parsed"), dict)
            and row.get("parsed", {}).get("answer") is not None
        }
        if len(answer_fingerprints) > 1:
            categories.append("answer_changes_across_positions")
        if not categories:
            continue
        base = {
            "scope": "group",
            "run": run,
            "group_id_sha256": fingerprint(group_id),
            "sample_id_sha256": [
                fingerprint(str(row["sample_id"])) for row in group
            ],
            "answer_correct_by_position": [
                bool(row["answer_correct"]) for row in group
            ],
            "context": safe_context(group[0]),
        }
        for category in categories:
            yield {"category": category, **base}


def cross_run_examples(
    runs: dict[str, list[dict[str, Any]]]
) -> Iterable[dict[str, Any]]:
    if "base" not in runs:
        return
    base_by_id = {str(row["sample_id"]): row for row in runs["base"]}
    for run, rows in sorted(runs.items()):
        if run == "base":
            continue
        for row in rows:
            sample_id = str(row["sample_id"])
            base = base_by_id.get(sample_id)
            if base is None:
                continue
            base_scored = scored_flags(base)
            treatment_scored = scored_flags(row)
            categories: list[str] = []
            if bool(base["answer_correct"]) and not bool(row["answer_correct"]):
                categories.append("base_only_answer_success")
            if not bool(base["answer_correct"]) and bool(row["answer_correct"]):
                categories.append("treatment_only_answer_success")
            base_quote = base_scored["evidence_quotes_correct"]
            treatment_quote = treatment_scored["evidence_quotes_correct"]
            if base_quote is False and treatment_quote is True:
                categories.append("treatment_quote_recovery")
            for category in categories:
                yield {
                    "scope": "cross_run",
                    "category": category,
                    "base_run": "base",
                    "treatment_run": run,
                    "sample_id_sha256": fingerprint(sample_id),
                    "group_id_sha256": fingerprint(str(row["group_id"])),
                    "context": safe_context(row),
                    "base_flags": base_scored,
                    "treatment_flags": treatment_scored,
                }


def audit_denominators(runs: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    eligible_groups = 0
    for rows in runs.values():
        group_counts = Counter(str(row["group_id"]) for row in rows)
        eligible_groups += sum(count >= 3 for count in group_counts.values())
    cross_run_comparisons = 0
    if "base" in runs:
        base_ids = {str(row["sample_id"]) for row in runs["base"]}
        cross_run_comparisons = sum(
            str(row["sample_id"]) in base_ids
            for run, rows in runs.items()
            if run != "base"
            for row in rows
        )
    return {
        "row": sum(len(rows) for rows in runs.values()),
        "group": eligible_groups,
        "cross_run": cross_run_comparisons,
    }


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()
    if args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    try:
        parsed_runs = [parse_run(value) for value in args.run]
        names = [name for name, _ in parsed_runs]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate --run name")
        runs = {name: load_run(name, path) for name, path in parsed_runs}
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    candidates: list[dict[str, Any]] = []
    for run, rows in sorted(runs.items()):
        for row in rows:
            for category in row_categories(row):
                candidates.append(row_example(run, category, row))
        candidates.extend(group_examples(run, rows))
    candidates.extend(cross_run_examples(runs))
    candidates.sort(
        key=lambda item: (
            item["category"],
            item.get("run", item.get("treatment_run", "")),
            item.get("group_id_sha256", ""),
            str(item.get("sample_id_sha256", "")),
        )
    )
    counts = Counter(item["category"] for item in candidates)
    category_scopes: dict[str, str] = {}
    for item in candidates:
        category = item["category"]
        scope = item["scope"]
        previous = category_scopes.setdefault(category, scope)
        if previous != scope:
            raise SystemExit(
                f"Internal accounting error: {category} spans {previous} and {scope}"
            )
    denominators = audit_denominators(runs)
    category_rates = {
        category: counts[category] / denominators[scope]
        for category, scope in category_scopes.items()
        if denominators[scope] > 0
    }
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for item in candidates:
        if selected_counts[item["category"]] >= args.max_examples:
            continue
        selected.append(item)
        selected_counts[item["category"]] += 1

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_records = [
        {
            "run": name,
            "path": portable_source_path(path),
            "rows": len(runs[name]),
            "sha256": sha256_file(path),
        }
        for name, path in parsed_runs
    ]
    payload = {
        "schema_version": "license-safe-failure-case-catalog-v1",
        "status": "validated",
        "raw_text_policy": (
            "No prompt, question, target, generated text, parsed answer, evidence "
            "quote, benchmark answer, raw identifier, or absolute path is emitted; "
            "traceability uses SHA-256 fingerprints."
        ),
        "sources": source_records,
        "category_counts": dict(sorted(counts.items())),
        "category_scopes": dict(sorted(category_scopes.items())),
        "scope_denominators": denominators,
        "category_rates": dict(sorted(category_rates.items())),
        "max_examples_per_category": args.max_examples,
        "examples": selected,
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    for forbidden in TEXT_KEYS_FORBIDDEN:
        if f'"{forbidden}":' in json_text:
            raise SystemExit(f"Internal safety error: raw text key emitted: {forbidden}")
    json_path = output / "failure_case_catalog.json"
    atomic_write(json_path, json_text)

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=(
            "category",
            "scope",
            "run",
            "base_run",
            "treatment_run",
            "group_id_sha256",
            "sample_id_sha256",
        ),
    )
    writer.writeheader()
    for item in selected:
        rendered = {}
        for key in writer.fieldnames:
            value = item.get(key, "")
            rendered[key] = (
                json.dumps(value, separators=(",", ":"))
                if isinstance(value, list)
                else value
            )
        writer.writerow(rendered)
    atomic_write(output / "failure_case_catalog.csv", csv_buffer.getvalue())

    markdown = [
        "# License-safe failure case catalog",
        "",
        payload["raw_text_policy"],
        "",
        "| Category | Scope | Candidates | Denominator | Rate | Selected fingerprints |",
        "|---|---|---:|---:|---:|---|",
    ]
    for category, count in sorted(counts.items()):
        identifiers = []
        for item in selected:
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
        scope = category_scopes[category]
        denominator = denominators[scope]
        rate = category_rates[category]
        markdown.append(
            f"| `{category}` | `{scope}` | {count} | {denominator} | "
            f"{rate:.6f} | {', '.join(identifiers)} |"
        )
    markdown.append("")
    atomic_write(output / "failure_case_catalog.md", "\n".join(markdown))

    manifest = {
        "schema_version": "failure-case-catalog-manifest-v1",
        "status": "validated",
        "source_sha256": {item["run"]: item["sha256"] for item in source_records},
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                output / "failure_case_catalog.json",
                output / "failure_case_catalog.csv",
                output / "failure_case_catalog.md",
            )
        },
    }
    atomic_write(
        output / "failure_case_catalog.manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"Wrote {len(selected)} license-safe examples across {len(counts)} categories"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
