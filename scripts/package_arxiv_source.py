#!/usr/bin/env python3
"""Build a deterministic arXiv source archive after the submission audit passes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


ALLOWED_SUFFIXES = {".tex", ".bib", ".bbl", ".bst", ".sty", ".cls", ".pdf", ".png", ".jpg", ".jpeg", ".md"}
EXCLUDED_NAMES = {"arxiv-audit.json", "arxiv-audit.scaffold.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink is forbidden in arXiv source: {path}")
        if not path.is_file() or path.name in EXCLUDED_NAMES:
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError(f"Single arXiv source file exceeds 50 MiB: {path}")
        files.append(path)
    return files


def build_deterministic_archive(root: Path, files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    relative = path.relative_to(root)
                    data = path.read_bytes()
                    info = tarfile.TarInfo(str(relative))
                    info.size = len(data)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))
    temporary.replace(output)


def validate_evidence_references(
    evidence_path: Path, evidence: dict[str, object]
) -> None:
    """Re-hash every project evidence file before trusting final_release_ready."""
    project_root = (
        evidence_path.parent.parent
        if evidence_path.parent.name == "results"
        else evidence_path.parent
    ).resolve()
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
        if (
            not path.is_file()
            or path.stat().st_size != int(raw_record.get("bytes", -1))
            or sha256_file(path) != raw_record.get("sha256")
        ):
            raise SystemExit(f"Evidence file changed after final audit: {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output-tar", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.paper_dir.resolve()
    evidence_path = args.evidence_manifest.resolve()
    if not evidence_path.is_file():
        raise SystemExit("Final full-paper evidence manifest is missing")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("schema_version") != "full-paper-evidence-manifest-v1"
        or evidence.get("status") != "validated"
        or evidence.get("final_release_ready") is not True
        or "compute_accounting" not in evidence.get("required_evidence_labels", [])
        or "compute_accounting" not in evidence.get("evidence", {})
    ):
        raise SystemExit("Full-paper evidence manifest is not final-release ready")
    validate_evidence_references(evidence_path, evidence)
    with tempfile.TemporaryDirectory() as directory:
        audit_path = Path(directory) / "audit.json"
        audit_script = Path(__file__).with_name("audit_arxiv_source.py")
        completed = subprocess.run(
            [
                "python3",
                str(audit_script),
                "--paper-dir",
                str(root),
                "--output",
                str(audit_path),
            ],
            text=True,
            capture_output=True,
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or audit.get("status") != "passed":
            raise SystemExit(
                "Submission audit has not passed: "
                f"errors={len(audit.get('errors', []))} pending={len(audit.get('pending', []))}"
            )

    try:
        files = selected_files(root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    required = {"main.tex", "main.bbl", "generated/results.tex"}
    relative_names = {str(path.relative_to(root)) for path in files}
    missing = required - relative_names
    if missing:
        raise SystemExit("Final source selection is missing: " + ", ".join(sorted(missing)))
    build_deterministic_archive(root, files, args.output_tar)
    manifest = {
        "schema_version": "arxiv-package-manifest-v1",
        "status": "validated",
        "source_audit_status": "passed",
        "full_evidence_manifest": {
            "filename": evidence_path.name,
            "sha256": sha256_file(evidence_path),
            "final_release_ready": True,
        },
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "archive": {
            "path": str(args.output_tar.resolve()),
            "bytes": args.output_tar.stat().st_size,
            "sha256": sha256_file(args.output_tar),
            "deterministic_metadata": True,
        },
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Packaged {len(files)} audited arXiv source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
