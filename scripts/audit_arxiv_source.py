#!/usr/bin/env python3
"""Audit an arXiv source tree for completeness, traceability, and secret hygiene."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FILES = ("main.tex", "references.bib", "generated/results.tex", "README.md")
SECRET_PATTERNS = {
    "ssh_password_label": re.compile(r"(?:password|密码)\s*[:：=]", re.IGNORECASE),
    "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
    "autodl_ssh_command": re.compile(r"ssh\s+-p\s+\d+\s+root@connect\.", re.IGNORECASE),
}
ABSOLUTE_PATH_PATTERNS = {
    "unix_home": re.compile(r"/(?:home|root)/[^\s{}]+"),
    "windows_drive": re.compile(r"[A-Za-z]:\\[^\s{}]+"),
}
CITATION_PATTERN = re.compile(r"\\cite[pt]?(?:\[[^\]]*\])?\{([^}]+)\}")
BIB_KEY_PATTERN = re.compile(r"@[A-Za-z]+\s*\{\s*([^,\s]+)")
GRAPHIC_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
LABEL_PATTERN = re.compile(r"\\label\{([^}]+)\}")
REFERENCE_PATTERN = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_issue(issues: list[dict[str, Any]], code: str, message: str, file: str | None = None) -> None:
    issue = {"code": code, "message": message}
    if file is not None:
        issue["file"] = file
    issues.append(issue)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Validate a work-in-progress scaffold without treating pending text or figures as failure.",
    )
    args = parser.parse_args()
    root = args.paper_dir.resolve()
    errors: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            add_issue(errors, "missing_required_file", f"Missing or empty required file: {relative}")
            continue
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if errors:
        payload = {
            "schema_version": "arxiv-source-audit-v1",
            "status": "failed",
            "paper_dir": str(root),
            "files": files,
            "errors": errors,
            "pending": pending,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    text_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in {".tex", ".bib", ".md"}
    )
    texts = {str(path.relative_to(root)): path.read_text(encoding="utf-8") for path in text_files}
    for relative, text in texts.items():
        if "[PENDING:" in text or "\\pending{" in text:
            add_issue(pending, "pending_content", "Unresolved PENDING content", relative)
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                add_issue(errors, f"secret_{name}", f"Possible credential material: {name}", relative)
        for name, pattern in ABSOLUTE_PATH_PATTERNS.items():
            if pattern.search(text):
                add_issue(errors, f"absolute_path_{name}", f"Absolute local path found: {name}", relative)

    main_text = texts["main.tex"]
    bib_text = texts["references.bib"]
    cited = {
        key.strip()
        for match in CITATION_PATTERN.finditer(main_text)
        for key in match.group(1).split(",")
        if key.strip()
    }
    defined = set(BIB_KEY_PATTERN.findall(bib_text))
    for key in sorted(cited - defined):
        add_issue(errors, "undefined_citation", f"Citation key is not defined: {key}", "main.tex")
    for key in sorted(defined - cited):
        add_issue(pending, "unused_bibliography_entry", f"Bibliography key is unused: {key}", "references.bib")

    labels = LABEL_PATTERN.findall(main_text)
    references = set(REFERENCE_PATTERN.findall(main_text))
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    for label in duplicate_labels:
        add_issue(errors, "duplicate_label", f"LaTeX label is defined more than once: {label}", "main.tex")
    for label in sorted(references - set(labels)):
        add_issue(errors, "undefined_reference", f"LaTeX reference is not defined: {label}", "main.tex")

    graphics = []
    for match in GRAPHIC_PATTERN.finditer(main_text):
        relative = match.group(1)
        graphics.append(relative)
        if not (root / relative).is_file():
            add_issue(pending, "missing_graphic", f"Referenced graphic is missing: {relative}", "main.tex")
    if "author names" in main_text.lower() or "author metadata" in main_text.lower():
        add_issue(pending, "author_metadata", "Author metadata has not been finalized", "main.tex")
    bbl_path = root / "main.bbl"
    if not bbl_path.is_file() or not bbl_path.stat().st_size:
        add_issue(pending, "missing_bbl", "Compiled bibliography main.bbl is missing", "main.bbl")
    else:
        files["main.bbl"] = {
            "bytes": bbl_path.stat().st_size,
            "sha256": sha256_file(bbl_path),
        }

    blocking_pending = [] if args.allow_pending else pending
    status = "passed" if not errors and not blocking_pending else "failed"
    payload = {
        "schema_version": "arxiv-source-audit-v1",
        "status": status,
        "mode": "scaffold" if args.allow_pending else "submission",
        "paper_dir": str(root),
        "files": files,
        "citations": {"cited": sorted(cited), "defined": sorted(defined)},
        "graphics": graphics,
        "cross_references": {
            "labels": sorted(set(labels)),
            "references": sorted(references),
        },
        "errors": errors,
        "pending": pending,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"arXiv source audit: status={status} errors={len(errors)} "
        f"pending={len(pending)} mode={payload['mode']}"
    )
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
