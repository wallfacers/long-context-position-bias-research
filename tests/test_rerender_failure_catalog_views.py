from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "scripts" / "analyze_failure_cases.py"
AUDIT = ROOT / "scripts" / "audit_failure_case_catalogs.py"
RERENDER = ROOT / "scripts" / "rerender_failure_catalog_views.py"


def build_catalog(project: Path) -> tuple[Path, Path]:
    source = project / "results" / "suite" / "base.jsonl"
    source.parent.mkdir(parents=True)
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
    directory = source.parent / "failure_cases"
    subprocess.run(
        [
            "python3",
            str(ANALYZE),
            "--run",
            f"base={source}",
            "--output-dir",
            str(directory),
        ],
        check=True,
        cwd=project,
        capture_output=True,
        text=True,
    )
    return source, directory


def run_audit(project: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(AUDIT),
            "--project-root",
            str(project),
            "--results-root",
            str(project / "results"),
            "--expected-catalogs",
            "1",
            "--output",
            str(project / "results" / "failure_case_catalog_audit.json"),
        ],
        check=False,
        cwd=project,
        capture_output=True,
        text=True,
    )


def test_rerender_repairs_only_views_and_manifest(tmp_path: Path):
    source, directory = build_catalog(tmp_path)
    json_path = directory / "failure_case_catalog.json"
    manifest_path = directory / "failure_case_catalog.manifest.json"
    json_before = json_path.read_bytes()
    source_before = source.read_bytes()
    for name, suffix in (
        ("failure_case_catalog.csv", "bad,row,['deadbeef']\n"),
        (
            "failure_case_catalog.md",
            "| `bad` | `group` | 1 | 1 | 1.000000 | ['deadbeef'] |\n",
        ),
    ):
        path = directory / name
        path.write_text(path.read_text(encoding="utf-8") + suffix, encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    for name in ("failure_case_catalog.csv", "failure_case_catalog.md"):
        path = directory / name
        manifest["outputs"][name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    assert run_audit(tmp_path).returncode != 0

    result = subprocess.run(
        ["python3", str(RERENDER), "--catalog-dir", str(directory)],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json_path.read_bytes() == json_before
    assert source.read_bytes() == source_before
    assert "['" not in (directory / "failure_case_catalog.csv").read_text()
    assert "['" not in (directory / "failure_case_catalog.md").read_text()
    assert run_audit(tmp_path).returncode == 0


def test_rerender_rejects_unsafe_json_without_changing_views(tmp_path: Path):
    _, directory = build_catalog(tmp_path)
    json_path = directory / "failure_case_catalog.json"
    catalog = json.loads(json_path.read_text())
    catalog["examples"][0]["answer"] = "must-not-publish"
    json_path.write_text(json.dumps(catalog) + "\n", encoding="utf-8")
    before = {
        name: (directory / name).read_bytes()
        for name in ("failure_case_catalog.csv", "failure_case_catalog.md")
    }
    result = subprocess.run(
        ["python3", str(RERENDER), "--catalog-dir", str(directory)],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "forbidden raw keys" in result.stderr
    assert before == {name: (directory / name).read_bytes() for name in before}
