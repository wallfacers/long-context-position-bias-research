#!/usr/bin/env python3
"""Build a secret-safe, license-conscious public reproducibility directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from sanitize_release_tree import TEXT_SUFFIXES, sanitize_text, sha256_file


ROOT_FILES = {
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements-eval.txt",
    "requirements-test.txt",
    "requirements-train.txt",
}
SOURCE_DIRS = {"configs", "docs", "paper", "scripts", "src", "tests"}
SOURCE_SUFFIXES = {
    ".bib",
    ".bbl",
    ".cls",
    ".csv",
    ".json",
    ".md",
    ".pdf",
    ".png",
    ".py",
    ".sh",
    ".sty",
    ".svg",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
DATA_SAFE_NAMES = {
    "chat_protocol_audit.json",
    "completion.json",
    "dataset_info.json",
    "manifest.json",
    "matched-audit.json",
    "matched-design.json",
    "pretokenization.json",
    "prompt_length_audit.json",
    "state.json",
}
RESULT_SAFE_SUFFIXES = {".csv", ".json", ".md", ".pdf", ".png", ".sha256", ".svg", ".txt"}
OUTPUT_SAFE_NAMES = {"CANARY_COMPLETE.json", "README.md", "run_config.json"}
OUTPUT_SAFE_SUFFIXES = {".csv", ".json", ".md", ".pdf", ".png", ".sha256", ".svg", ".txt"}
FINAL_REQUIRED_EVIDENCE_LABELS = {"compute_accounting"}


def validate_evidence_references(
    project_root: Path, evidence: dict[str, object]
) -> None:
    """Re-hash every project evidence file before trusting final_release_ready."""
    project_root = project_root.resolve()
    records = evidence.get("evidence", {})
    if not isinstance(records, dict) or not records:
        raise SystemExit("Full-paper evidence manifest has no evidence records")
    for label, raw_record in records.items():
        if not isinstance(raw_record, dict):
            raise SystemExit(f"Malformed evidence record: {label}")
        relative = raw_record.get("path")
        if not isinstance(relative, str) or not relative:
            raise SystemExit(f"Evidence record has no project-relative path: {label}")
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise SystemExit(f"Evidence record escapes the project tree: {label}") from exc
        try:
            expected_bytes = int(raw_record.get("bytes", -1))
        except (TypeError, ValueError):
            raise SystemExit(f"Malformed evidence byte count: {label}") from None
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != raw_record.get("sha256")
        ):
            raise SystemExit(f"Evidence file changed after final audit: {label}")


def is_safe_data_file(path: Path) -> bool:
    return (
        path.name in DATA_SAFE_NAMES
        or path.name.endswith(".manifest.json")
        or path.name.endswith(".audit.json")
    )


def is_safe_result_file(path: Path) -> bool:
    if path.name.endswith(".jsonl.gz"):
        return "bootstrap_indices" in path.name
    return path.suffix.lower() in RESULT_SAFE_SUFFIXES


def is_safe_output_file(path: Path) -> bool:
    if path.name in OUTPUT_SAFE_NAMES:
        return True
    return "training_diagnostics" in path.parts and path.suffix.lower() in OUTPUT_SAFE_SUFFIXES


def selected_files(root: Path) -> tuple[list[Path], Counter[str]]:
    selected = []
    excluded: Counter[str] = Counter()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        top = relative.parts[0]
        include = False
        if len(relative.parts) == 1:
            include = path.name in ROOT_FILES or path.name.upper().startswith("LICENSE")
        elif top in SOURCE_DIRS:
            include = path.suffix.lower() in SOURCE_SUFFIXES and "__pycache__" not in path.parts
        elif top == "data":
            include = is_safe_data_file(path)
        elif top == "results":
            include = is_safe_result_file(path)
        elif top == "outputs":
            include = is_safe_output_file(path)
        elif top == "third_party":
            include = (
                path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
                or path.suffix.lower() == ".sha256"
                or path.name == "frozen-source-download-manifest.json"
            )
        if path.is_symlink():
            if include or top in SOURCE_DIRS:
                raise ValueError(f"Symlink is forbidden in selected public source: {path}")
            excluded[top] += 1
            continue
        if not path.is_file():
            continue
        if include:
            selected.append(path)
        else:
            excluded[top] += 1
    return selected, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preflight-allow-incomplete-evidence",
        action="store_true",
        help="Mark a temporary sanitizer/test rehearsal; forbidden for a final release.",
    )
    args = parser.parse_args()
    root, evidence, output = (
        args.project_root.resolve(),
        args.evidence_manifest.resolve(),
        args.output.resolve(),
    )
    if not root.is_dir() or not evidence.is_file():
        raise SystemExit("Project root and full evidence manifest are required")
    try:
        evidence.relative_to(root)
    except ValueError as exc:
        raise SystemExit("Full evidence manifest must be inside the project tree") from exc
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    if (
        evidence_payload.get("schema_version") != "full-paper-evidence-manifest-v1"
        or evidence_payload.get("status") != "validated"
    ):
        raise SystemExit("Full paper evidence manifest has not passed")
    required_labels = set(evidence_payload.get("required_evidence_labels", []))
    evidence_labels = set(evidence_payload.get("evidence", {}))
    missing_final_labels = FINAL_REQUIRED_EVIDENCE_LABELS - (
        required_labels & evidence_labels
    )
    final_release_ready = evidence_payload.get("final_release_ready") is True
    if (
        (missing_final_labels or not final_release_ready)
        and not args.preflight_allow_incomplete_evidence
    ):
        raise SystemExit(
            "Full paper manifest is not final-release ready; missing/undeclared evidence: "
            + ", ".join(sorted(missing_final_labels or FINAL_REQUIRED_EVIDENCE_LABELS))
        )
    if final_release_ready:
        validate_evidence_references(root, evidence_payload)
    if output.exists():
        raise SystemExit("--output already exists; public release generation never overwrites")
    if output == root or root in output.parents:
        raise SystemExit("--output cannot be inside the project tree")
    try:
        files, excluded = selected_files(root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output.mkdir(parents=True)
    records = []
    try:
        for path in files:
            relative = path.relative_to(root)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            replacements = 0
            if path.suffix.lower() in TEXT_SUFFIXES:
                text, replacements = sanitize_text(path.read_text(encoding="utf-8"))
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
        policy = {
            "schema_version": "public-release-selection-v1",
            "status": "validated",
            "full_evidence_manifest_sha256": sha256_file(evidence),
            "evidence_completeness_preflight_bypass": bool(
                args.preflight_allow_incomplete_evidence
            ),
            "selected_files": len(records),
            "excluded_file_counts_by_top_level": dict(sorted(excluded.items())),
            "raw_jsonl_policy": (
                "Raw benchmark/training/generation JSONL is excluded by default. Only "
                "compressed bootstrap index JSONL is public; benchmark text is rebuilt "
                "from pinned sources under its original license."
            ),
            "weights_policy": "Base weights, adapters, checkpoints, and partial files are excluded.",
        }
        policy_path = output / "public_release_selection.json"
        policy_path.write_text(
            json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "public-release-manifest-v1",
            "status": "validated",
            "selection_policy_sha256": sha256_file(policy_path),
            "files": records,
            "files_with_path_replacements": sum(
                record["path_replacements"] > 0 for record in records
            ),
            "secret_policy": "fail closed; credentials are never silently redacted",
        }
        manifest_path = output / "public_release_manifest.json"
        temporary = manifest_path.with_name(manifest_path.name + f".tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    except Exception:
        shutil.rmtree(output)
        raise
    print(f"Built public release with {len(records)} selected files at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
