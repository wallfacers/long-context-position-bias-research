from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "scripts" / "analyze_failure_cases.py"
REPACK = ROOT / "scripts" / "repack_failure_catalog_artifact.sh"


def prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    result = project / "results" / "suite"
    catalogs = []
    for seed in (1, 2):
        source = result / f"seed_{seed}.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            json.dumps(
                {
                    "sample_id": f"sample-{seed}",
                    "group_id": f"group-{seed}",
                    "task": "synthetic_test",
                    "valid_json": True,
                    "answer_correct": False,
                    "evidence_quotes_correct": False,
                }
            )
            + "\n"
        )
        output = result / f"failure_cases_seed_{seed}"
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
        catalogs.append(
            (output / "failure_case_catalog.manifest.json")
            .relative_to(result)
            .as_posix()
        )
    (result / "completion.json").write_text(
        json.dumps(
            {
                "schema_version": "multiseed-v1",
                "status": "validated",
                "failure_case_catalogs": catalogs,
            }
        )
        + "\n"
    )
    (result / "RESULTS_READY_FOR_AGENT_REVIEW").touch()
    artifact = tmp_path / "suite.tar.gz"
    with tarfile.open(artifact, "w:gz") as archive:
        archive.add(result, arcname="results/suite")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    Path(str(artifact) + ".sha256").write_text(f"{digest}  {artifact}\n")
    return project, result, artifact


def command(project: Path, result: Path, artifact: Path) -> list[str]:
    return [
        "bash",
        str(REPACK),
        "--project-root",
        str(project),
        "--result-dir",
        str(result),
        "--artifact",
        str(artifact),
        "--expected-catalogs",
        "2",
    ]


def test_repack_audits_multiseed_catalogs_and_updates_checksum(tmp_path: Path):
    project, result, artifact = prepare(tmp_path)
    before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    run = subprocess.run(
        command(project, result, artifact),
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    after = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert after != before
    assert Path(str(artifact) + ".sha256").read_text().split()[0] == after
    audit = json.loads((result / "failure_case_catalog_audit.json").read_text())
    assert audit["status"] == "validated"
    assert audit["expected_catalogs"] == 2
    with tarfile.open(artifact) as archive:
        assert "results/suite/failure_case_catalog_audit.json" in archive.getnames()


def test_repack_leaves_artifact_unchanged_when_a_view_is_noncanonical(tmp_path: Path):
    project, result, artifact = prepare(tmp_path)
    markdown = result / "failure_cases_seed_1" / "failure_case_catalog.md"
    markdown.write_text(markdown.read_text() + "| bad | row | 1 | 1 | 1 | ['bad'] |\n")
    manifest_path = markdown.with_name("failure_case_catalog.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][markdown.name] = {
        "bytes": markdown.stat().st_size,
        "sha256": hashlib.sha256(markdown.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest) + "\n")
    before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    run = subprocess.run(
        command(project, result, artifact),
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "Markdown is not the canonical JSON-derived view" in run.stderr
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before
    assert not (result / "failure_case_catalog_audit.json").exists()
