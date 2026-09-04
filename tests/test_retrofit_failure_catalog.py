from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "scripts" / "analyze_failure_cases.py"
RETROFIT = ROOT / "scripts" / "retrofit_failure_catalog.sh"


def test_retrofit_updates_completion_and_repackages_atomically(tmp_path: Path):
    project = tmp_path / "project"
    result = project / "results" / "suite"
    result.mkdir(parents=True)
    source = result / "base.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "group_id": "group",
                "task": "synthetic_test",
                "valid_json": True,
                "answer_correct": False,
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
            str(result / "failure_cases"),
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    (result / "completion.json").write_text(
        json.dumps({"schema_version": "suite-v1", "status": "validated"}) + "\n"
    )
    (result / "RESULTS_READY_FOR_AGENT_REVIEW").touch()
    artifact = tmp_path / "suite.tar.gz"
    with tarfile.open(artifact, "w:gz") as archive:
        archive.add(result, arcname="results/suite")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    Path(str(artifact) + ".sha256").write_text(f"{checksum}  {artifact}\n")

    subprocess.run(
        [
            "bash",
            str(RETROFIT),
            "--project-root",
            str(project),
            "--result-dir",
            str(result),
            "--artifact",
            str(artifact),
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    completion = json.loads((result / "completion.json").read_text())
    assert completion["failure_case_catalog"] == (
        "failure_cases/failure_case_catalog.manifest.json"
    )
    recorded = Path(str(artifact) + ".sha256").read_text().split()[0]
    assert recorded == hashlib.sha256(artifact.read_bytes()).hexdigest()
    with tarfile.open(artifact) as archive:
        names = set(archive.getnames())
    assert (
        "results/suite/failure_cases/failure_case_catalog.manifest.json" in names
    )

    completion_before = (result / "completion.json").read_bytes()
    artifact_before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path = result / "failure_cases" / "failure_case_catalog.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "invalid"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    failed = subprocess.run(
        [
            "bash",
            str(RETROFIT),
            "--project-root",
            str(project),
            "--result-dir",
            str(result),
            "--artifact",
            str(artifact),
        ],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert (result / "completion.json").read_bytes() == completion_before
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == artifact_before


def test_retrofit_rejects_rehashed_noncanonical_view_before_repack(tmp_path: Path):
    project = tmp_path / "project"
    result = project / "results" / "suite"
    result.mkdir(parents=True)
    source = result / "base.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "group_id": "group",
                "task": "synthetic_test",
                "valid_json": True,
                "answer_correct": False,
                "evidence_quotes_correct": False,
            }
        )
        + "\n"
    )
    subprocess.run(
        [
            "python3",
            str(ANALYZE),
            "--run",
            f"base={source}",
            "--output-dir",
            str(result / "failure_cases"),
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    (result / "completion.json").write_text(
        json.dumps({"schema_version": "suite-v1", "status": "validated"}) + "\n"
    )
    (result / "RESULTS_READY_FOR_AGENT_REVIEW").touch()
    artifact = tmp_path / "suite.tar.gz"
    with tarfile.open(artifact, "w:gz") as archive:
        archive.add(result, arcname="results/suite")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    Path(str(artifact) + ".sha256").write_text(f"{checksum}  {artifact}\n")

    directory = result / "failure_cases"
    markdown = directory / "failure_case_catalog.md"
    markdown.write_text(
        markdown.read_text() + "| `bad` | `group` | 1 | 1 | 1.000000 | ['bad'] |\n"
    )
    manifest_path = directory / "failure_case_catalog.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][markdown.name] = {
        "bytes": markdown.stat().st_size,
        "sha256": hashlib.sha256(markdown.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest) + "\n")
    artifact_before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    completion_before = (result / "completion.json").read_bytes()

    failed = subprocess.run(
        [
            "bash",
            str(RETROFIT),
            "--project-root",
            str(project),
            "--result-dir",
            str(result),
            "--artifact",
            str(artifact),
        ],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "Markdown is not the canonical JSON-derived view" in failed.stderr
    assert (result / "completion.json").read_bytes() == completion_before
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == artifact_before
