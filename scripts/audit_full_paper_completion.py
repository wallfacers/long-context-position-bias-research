#!/usr/bin/env python3
"""Validate and index the complete paper evidence package without repacking it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FINAL_RELEASE_EVIDENCE_LABELS = {"compute_accounting"}
EXPECTED_PRIMARY_STATUSES = {
    "Qwen2.5-7B": ["corrective"],
    "Mistral-7B-v0.3": ["confirmatory"],
}
JSON_EVIDENCE_SCHEMAS = {
    "experiment_completion": "strict-block96-experiment-completion-v1",
    "failure_case_catalog_audit": "failure-case-catalog-audit-v1",
    "compute_accounting": "active-gpu-compute-accounting-v1",
    "qwen_fixed100_realized_subset_audit": "realized-training-subset-audit-v1",
    "qwen_block96_realized_subset_audit": "realized-training-subset-audit-v1",
    "mistral_block96_realized_subset_audit": "realized-training-subset-audit-v1",
    "cross_family_rule": "seed-level-analysis-v1",
    "cross_family_nolima": "seed-level-analysis-v1",
    "cross_family_longbench": "seed-level-analysis-v1",
    "qwen_mmlu": "general-regression-analysis-v1",
    "qwen_ifeval": "ifeval-regression-analysis-v1",
    "qwen_mechanisms": "nolima-mechanism-analysis-v1",
    "mistral_mmlu": "general-regression-analysis-v1",
    "mistral_ifeval": "ifeval-regression-analysis-v1",
    "mistral_mechanisms": "nolima-mechanism-analysis-v1",
    "paper_results_manifest": "paper-results-generation-v1",
    "paper_figure_manifest": "seed-level-factorial-figure-v2",
}
PAPER_RESULT_SOURCE_LABELS = {
    "rule": "cross_family_rule",
    "nolima": "cross_family_nolima",
    "longbench": "cross_family_longbench",
    "qwen_mmlu": "qwen_mmlu",
    "qwen_ifeval": "qwen_ifeval",
    "qwen_mechanisms": "qwen_mechanisms",
    "mistral_mmlu": "mistral_mmlu",
    "mistral_ifeval": "mistral_ifeval",
    "mistral_mechanisms": "mistral_mechanisms",
}
PAPER_FIGURE_FILE_LABELS = {
    "pdf": "paper_figure_pdf",
    "svg": "paper_figure_svg",
    "png": "paper_figure_png",
    "table": "paper_figure_csv",
    "alt_text": "paper_figure_alt",
}


def labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(raw_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_unique(entries: list[tuple[str, Path]], kind: str) -> None:
    labels = [label for label, _ in entries]
    if len(labels) != len(set(labels)):
        raise SystemExit(f"Duplicate {kind} labels")


def audit_artifact(label: str, path: Path) -> dict[str, object]:
    path = path.resolve()
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit(f"Missing artifact {label}: {path}")
    if not checksum.is_file():
        raise SystemExit(f"Missing artifact checksum {label}: {checksum}")
    expected = checksum.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise SystemExit(f"Artifact checksum mismatch: {label}")
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": actual,
        "checksum_filename": checksum.name,
        "checksum_sha256": sha256_file(checksum),
    }


def validate_evidence_semantics(label: str, path: Path) -> None:
    expected_schema = JSON_EVIDENCE_SCHEMAS.get(label)
    if expected_schema is None:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Evidence is not valid JSON: {label}") from exc
    if payload.get("schema_version") != expected_schema:
        raise SystemExit(
            f"Evidence schema mismatch: {label} expected={expected_schema} "
            f"actual={payload.get('schema_version')}"
        )
    if label in {
        "experiment_completion",
        "failure_case_catalog_audit",
        "compute_accounting",
        "qwen_fixed100_realized_subset_audit",
        "qwen_block96_realized_subset_audit",
        "mistral_block96_realized_subset_audit",
        "paper_results_manifest",
        "paper_figure_manifest",
    } and payload.get("status") != "validated":
        raise SystemExit(f"Evidence is not validated: {label}")

    if label == "experiment_completion":
        if payload.get("historical_fixed100_primary_eligible") is not False:
            raise SystemExit("Experiment completion admits historical fixed-100 evidence")
    elif label == "failure_case_catalog_audit":
        if payload.get("expected_catalogs") != 18 or len(payload.get("catalogs", [])) != 18:
            raise SystemExit("Strict failure-catalog audit must contain exactly 18 catalogs")
    elif label == "compute_accounting":
        by_kind = payload.get("by_kind", {})
        if (
            payload.get("expected_training_step") != 96
            or float(payload.get("hourly_rate_cny", 0)) <= 0
            or int(by_kind.get("training", {}).get("events", 0)) <= 0
            or int(by_kind.get("evaluation", {}).get("events", 0)) <= 0
            or float(payload.get("total_active_gpu_seconds", 0)) <= 0
        ):
            raise SystemExit("Compute accounting is not a nonempty strict checkpoint-96 ledger")
    elif label.endswith("realized_subset_audit"):
        assessment = payload.get("claim_assessment", {})
        strict = assessment.get("strict_realized_fixed_step_matching_all_seeds")
        action = assessment.get("recommended_action")
        if label == "qwen_fixed100_realized_subset_audit":
            if strict is not False or action != "retrain_from_materialized_block_complete_subsets":
                raise SystemExit("Historical fixed-100 audit no longer fails the strict claim")
        elif strict is not True or action != "retain_current_runs":
            raise SystemExit(f"Strict realized-subset audit did not pass: {label}")
    elif label.startswith("cross_family_"):
        expected_kind = "natural_transfer" if label == "cross_family_longbench" else "factorial"
        if (
            payload.get("analysis_kind") != expected_kind
            or payload.get("primary_training_seed_summary") is not True
            or payload.get("confirmatory_only_primary_summary") is not False
            or payload.get("primary_statuses_by_family") != EXPECTED_PRIMARY_STATUSES
            or set(payload.get("families", {})) != set(EXPECTED_PRIMARY_STATUSES)
        ):
            raise SystemExit(f"Cross-family primary designation gate failed: {label}")
    elif label == "paper_results_manifest":
        if (
            payload.get("primary_summaries_confirmatory_only") is not False
            or payload.get("corrective_plus_confirmatory_primary") is not True
            or payload.get("primary_statuses_by_family") != EXPECTED_PRIMARY_STATUSES
        ):
            raise SystemExit("Paper-results manifest mislabels the strict primary evidence")
    elif label == "paper_figure_manifest":
        if (
            payload.get("confirmatory_only") is not False
            or payload.get("corrective_plus_confirmatory_primary") is not True
            or payload.get("primary_statuses_by_family") != EXPECTED_PRIMARY_STATUSES
        ):
            raise SystemExit("Paper-figure manifest mislabels the strict primary evidence")


def audit_evidence(label: str, path: Path, project_root: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit(f"Missing evidence {label}: {path}")
    try:
        relative = path.relative_to(project_root)
    except ValueError as exc:
        raise SystemExit(f"Evidence must be inside the project tree: {path}") from exc
    validate_evidence_semantics(label, path)
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_manifest_record(
    record: object,
    actual_path: Path,
    label: str,
    project_root: Path,
    *,
    require_bytes: bool = False,
) -> None:
    if not isinstance(record, dict):
        raise SystemExit(f"Derived manifest lacks a record for {label}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"Derived manifest lacks a path for {label}")
    recorded_path = Path(raw_path)
    if not recorded_path.is_absolute():
        recorded_path = project_root / recorded_path
    if recorded_path.resolve() != actual_path.resolve():
        raise SystemExit(f"Derived manifest path mismatch: {label}")
    if record.get("sha256") != sha256_file(actual_path):
        raise SystemExit(f"Derived manifest hash mismatch: {label}")
    if require_bytes and int(record.get("bytes", -1)) != actual_path.stat().st_size:
        raise SystemExit(f"Derived manifest byte-count mismatch: {label}")


def validate_derived_manifest_bindings(
    entries: dict[str, Path], project_root: Path
) -> None:
    """Bind paper-facing manifests to the exact audited inputs and outputs."""
    if "paper_results_manifest" in entries:
        required = set(PAPER_RESULT_SOURCE_LABELS.values()) | {"paper_results_tex"}
        missing = required - set(entries)
        if missing:
            raise SystemExit(
                "Paper-results manifest binding lacks evidence: "
                + ", ".join(sorted(missing))
            )
        manifest = json.loads(
            entries["paper_results_manifest"].read_text(encoding="utf-8")
        )
        sources = manifest.get("sources", {})
        if not isinstance(sources, dict):
            raise SystemExit("Paper-results manifest has malformed sources")
        for source_name, evidence_label in PAPER_RESULT_SOURCE_LABELS.items():
            require_manifest_record(
                sources.get(source_name),
                entries[evidence_label],
                f"paper_results.sources.{source_name}",
                project_root,
            )
        exploratory = sources.get("qwen_exploratory_rule")
        if not isinstance(exploratory, dict):
            raise SystemExit("Paper-results manifest lacks its exploratory source")
        raw_exploratory_path = exploratory.get("path")
        if not isinstance(raw_exploratory_path, str) or not raw_exploratory_path:
            raise SystemExit("Paper-results exploratory source has no path")
        exploratory_path = Path(raw_exploratory_path)
        if not exploratory_path.is_absolute():
            exploratory_path = project_root / exploratory_path
        exploratory_path = exploratory_path.resolve()
        try:
            exploratory_path.relative_to(project_root)
        except ValueError as exc:
            raise SystemExit("Paper-results exploratory source escapes project") from exc
        if not exploratory_path.is_file():
            raise SystemExit("Paper-results exploratory source is missing")
        require_manifest_record(
            exploratory,
            exploratory_path,
            "paper_results.sources.qwen_exploratory_rule",
            project_root,
        )
        if manifest.get("output_tex_sha256") != sha256_file(entries["paper_results_tex"]):
            raise SystemExit("Paper-results TeX changed relative to its manifest")

    if "paper_figure_manifest" in entries:
        required = set(PAPER_FIGURE_FILE_LABELS.values()) | {"cross_family_nolima"}
        missing = required - set(entries)
        if missing:
            raise SystemExit(
                "Paper-figure manifest binding lacks evidence: "
                + ", ".join(sorted(missing))
            )
        manifest = json.loads(
            entries["paper_figure_manifest"].read_text(encoding="utf-8")
        )
        if manifest.get("analysis_sha256") != sha256_file(entries["cross_family_nolima"]):
            raise SystemExit("Paper figure does not bind the audited NoLiMa analysis")
        files = manifest.get("files", {})
        if not isinstance(files, dict):
            raise SystemExit("Paper-figure manifest has malformed files")
        for file_name, evidence_label in PAPER_FIGURE_FILE_LABELS.items():
            require_manifest_record(
                files.get(file_name),
                entries[evidence_label],
                f"paper_figure.files.{file_name}",
                project_root,
                require_bytes=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--artifact", action="append", type=labeled_path, default=[])
    parser.add_argument("--evidence", action="append", type=labeled_path, default=[])
    parser.add_argument("--require-evidence-label", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.artifact or not args.evidence:
        raise SystemExit("At least one artifact and one evidence file are required")
    require_unique(args.artifact, "artifact")
    require_unique(args.evidence, "evidence")
    required_evidence_labels = sorted(set(args.require_evidence_label))
    missing_required = set(required_evidence_labels) - {
        label for label, _ in args.evidence
    }
    if missing_required:
        raise SystemExit(
            "Missing required evidence labels: " + ", ".join(sorted(missing_required))
        )
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise SystemExit("--project-root must be a directory")
    artifacts = {
        label: audit_artifact(label, path) for label, path in args.artifact
    }
    evidence_paths = {label: path.resolve() for label, path in args.evidence}
    evidence = {
        label: audit_evidence(label, path, project_root)
        for label, path in args.evidence
    }
    validate_derived_manifest_bindings(evidence_paths, project_root)
    payload = {
        "schema_version": "full-paper-evidence-manifest-v1",
        "status": "validated",
        "artifact_count": len(artifacts),
        "evidence_file_count": len(evidence),
        "required_evidence_labels": required_evidence_labels,
        "final_release_ready": FINAL_RELEASE_EVIDENCE_LABELS.issubset(
            set(required_evidence_labels) & set(evidence)
        ),
        "artifact_total_bytes": sum(int(row["bytes"]) for row in artifacts.values()),
        "artifacts": artifacts,
        "evidence": evidence,
        "integrity_policy": (
            "Every packaged suite must match its adjacent SHA-256 file; paper-facing "
            "evidence is hashed directly from the project tree."
        ),
    }
    output = args.output.resolve()
    try:
        output.relative_to(project_root)
    except ValueError as exc:
        raise SystemExit("--output must be inside the project tree") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        f"Validated {len(artifacts)} suite packages and {len(evidence)} paper evidence files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
