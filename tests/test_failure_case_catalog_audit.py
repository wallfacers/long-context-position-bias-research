from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "scripts" / "analyze_failure_cases.py"
AUDIT = ROOT / "scripts" / "audit_failure_case_catalogs.py"


def write_source(path: Path, prefix: str) -> None:
    rows = []
    for index, correct in enumerate((True, False, True)):
        rows.append(
            {
                "sample_id": f"{prefix}-s{index}",
                "group_id": f"{prefix}-group",
                "task": "synthetic_test",
                "position_label": f"p{index}",
                "target_position": index / 2,
                "valid_json": True,
                "answer_correct": correct,
                "evidence_quotes_correct": correct,
                "generated_text": f"private-{prefix}-{index}",
                "parsed": {"answer": f"private-answer-{index}"},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def build_catalog(project: Path, name: str) -> Path:
    source = project / "results" / name / "base.jsonl"
    output = project / "results" / name / "failure_cases"
    write_source(source, name)
    subprocess.run(
        [
            "python3",
            str(ANALYZE),
            "--run",
            f"base={source}",
            "--output-dir",
            str(output),
        ],
        check=True,
        cwd=project,
        capture_output=True,
        text=True,
    )
    return source


def run_audit(project: Path, expected: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(AUDIT),
            "--project-root",
            str(project),
            "--results-root",
            str(project / "results"),
            "--expected-catalogs",
            str(expected),
            "--output",
            str(project / "results" / "failure_case_catalog_audit.json"),
        ],
        check=False,
        cwd=project,
        capture_output=True,
        text=True,
    )


def test_audit_revalidates_every_catalog_and_source(tmp_path: Path):
    first = build_catalog(tmp_path, "first")
    build_catalog(tmp_path, "second")
    result = run_audit(tmp_path, 2)
    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (tmp_path / "results" / "failure_case_catalog_audit.json").read_text()
    )
    assert payload["status"] == "validated"
    assert payload["expected_catalogs"] == 2
    assert len(payload["catalogs"]) == 2
    assert payload["total_source_rows"] == 6

    first.write_text(first.read_text() + first.read_text().splitlines()[0] + "\n")
    changed = run_audit(tmp_path, 2)
    assert changed.returncode != 0
    assert "source hash mismatch" in changed.stderr


def test_audit_rejects_missing_expected_catalog(tmp_path: Path):
    build_catalog(tmp_path, "only")
    result = run_audit(tmp_path, 2)
    assert result.returncode != 0
    assert "Expected 2 failure catalogs, found 1" in result.stderr


def test_explicit_manifest_mode_ignores_unrelated_historical_catalogs(tmp_path: Path):
    build_catalog(tmp_path, "strict_a")
    build_catalog(tmp_path, "strict_b")
    build_catalog(tmp_path, "historical")
    manifests = [
        tmp_path / "results" / name / "failure_cases" / "failure_case_catalog.manifest.json"
        for name in ("strict_a", "strict_b")
    ]
    command = [
        "python3",
        str(AUDIT),
        "--project-root",
        str(tmp_path),
    ]
    for manifest in manifests:
        command.extend(("--manifest", str(manifest)))
    command.extend(
        (
            "--expected-catalogs",
            "2",
            "--output",
            str(tmp_path / "results" / "strict_audit.json"),
        )
    )
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "results" / "strict_audit.json").read_text())
    assert len(payload["catalogs"]) == 2
    assert all("historical" not in row["manifest"] for row in payload["catalogs"])


def test_audit_rejects_raw_field_even_if_output_manifest_is_rehashed(tmp_path: Path):
    build_catalog(tmp_path, "suite")
    directory = tmp_path / "results" / "suite" / "failure_cases"
    catalog_path = directory / "failure_case_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["examples"][0]["answer"] = "must-not-publish"
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = directory / "failure_case_catalog.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][catalog_path.name] = {
        "bytes": catalog_path.stat().st_size,
        "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = run_audit(tmp_path, 1)
    assert result.returncode != 0
    assert "forbidden raw keys" in result.stderr


def test_audit_rejects_noncanonical_markdown_even_if_rehashed(tmp_path: Path):
    build_catalog(tmp_path, "suite")
    directory = tmp_path / "results" / "suite" / "failure_cases"
    markdown_path = directory / "failure_case_catalog.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8").replace(
            "synthetic_test", "synthetic_test"
        )
        + "| `bad` | `group` | 1 | 1 | 1.000000 | ['deadbeef'] |\n",
        encoding="utf-8",
    )
    manifest_path = directory / "failure_case_catalog.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][markdown_path.name] = {
        "bytes": markdown_path.stat().st_size,
        "sha256": hashlib.sha256(markdown_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = run_audit(tmp_path, 1)
    assert result.returncode != 0
    assert "Markdown is not the canonical JSON-derived view" in result.stderr


def test_audit_rejects_noncanonical_csv_even_if_rehashed(tmp_path: Path):
    build_catalog(tmp_path, "suite")
    directory = tmp_path / "results" / "suite" / "failure_cases"
    csv_path = directory / "failure_case_catalog.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8") + "bad,row,['deadbeef']\n",
        encoding="utf-8",
    )
    manifest_path = directory / "failure_case_catalog.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][csv_path.name] = {
        "bytes": csv_path.stat().st_size,
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = run_audit(tmp_path, 1)
    assert result.returncode != 0
    assert "CSV is not the canonical JSON-derived view" in result.stderr


def test_audit_accepts_a_valid_catalog_with_no_failures(tmp_path: Path):
    source = tmp_path / "results" / "perfect" / "base.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "group_id": "group",
                "task": "synthetic_test",
                "valid_json": True,
                "answer_correct": True,
                "evidence_quotes_applicable": False,
                "evidence_quotes_correct": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "python3",
            str(ANALYZE),
            "--run",
            f"base={source}",
            "--output-dir",
            str(source.parent / "failure_cases"),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    result = run_audit(tmp_path, 1)
    assert result.returncode == 0, result.stderr
