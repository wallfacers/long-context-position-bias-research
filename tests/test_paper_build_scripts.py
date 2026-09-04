import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_paper_pdf.sh"
INSTALL_SCRIPT = ROOT / "scripts" / "install_tectonic.sh"


def _paper_tree(path: Path) -> Path:
    paper = path / "paper"
    (paper / "generated").mkdir(parents=True)
    (paper / "main.tex").write_text("paper\n", encoding="utf-8")
    (paper / "references.bib").write_text("refs\n", encoding="utf-8")
    (paper / "generated" / "results.tex").write_text("results\n", encoding="utf-8")
    return paper


def _fake_tectonic(
    path: Path,
    *,
    undefined: bool = False,
    layout_warning: bool = False,
    missing_page_count: bool = False,
) -> Path:
    executable = path / "tectonic"
    warning = "LaTeX Warning: Citation missing undefined" if undefined else "clean"
    if layout_warning:
        warning = r"Overfull \hbox (2.0pt too wide)"
    page_line = (
        ""
        if missing_page_count
        else "printf 'Output written on main.xdv (3 pages, 42 bytes).\\n' >> \"$outdir/main.log\""
    )
    executable.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "--version" ]]; then
  echo 'Tectonic 0.17.0-test'
  exit 0
fi
outdir=''
while [[ $# -gt 0 ]]; do
  if [[ "$1" == '--outdir' ]]; then outdir="$2"; shift 2; else shift; fi
done
printf 'pdf' > "$outdir/main.pdf"
printf 'bbl' > "$outdir/main.bbl"
printf '%s\n' "{warning}" > "$outdir/main.log"
{page_line}
printf 'biblog\n' > "$outdir/main.blg"
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_explicit_tectonic_build_records_engine_and_hashes(tmp_path: Path):
    paper = _paper_tree(tmp_path)
    tectonic = _fake_tectonic(tmp_path)
    output = tmp_path / "artifacts" / "paper.pdf"
    manifest = tmp_path / "artifacts" / "build.json"

    subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--paper-dir",
            str(paper),
            "--output-pdf",
            str(output),
            "--build-manifest",
            str(manifest),
            "--engine",
            "tectonic",
            "--tectonic-bin",
            str(tectonic),
        ],
        check=True,
        cwd=ROOT,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "paper-pdf-build-v2"
    assert payload["status"] == "validated"
    assert payload["engine"] == {
        "name": "tectonic",
        "version": "Tectonic 0.17.0-test",
    }
    assert payload["undefined_references"] == 0
    assert payload["layout_box_warnings"] == 0
    assert payload["pdf"]["pages"] == 3
    assert output.read_bytes() == b"pdf"
    assert (paper / "main.bbl").read_bytes() == b"bbl"


def test_tectonic_build_rejects_undefined_citation(tmp_path: Path):
    paper = _paper_tree(tmp_path)
    tectonic = _fake_tectonic(tmp_path, undefined=True)
    result = subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--paper-dir",
            str(paper),
            "--output-pdf",
            str(tmp_path / "paper.pdf"),
            "--build-manifest",
            str(tmp_path / "build.json"),
            "--engine",
            "tectonic",
            "--tectonic-bin",
            str(tectonic),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "undefined citations or references" in result.stderr
    assert not (tmp_path / "paper.pdf").exists()


def test_tectonic_build_rejects_layout_box_warning(tmp_path: Path):
    paper = _paper_tree(tmp_path)
    tectonic = _fake_tectonic(tmp_path, layout_warning=True)
    result = subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--paper-dir",
            str(paper),
            "--output-pdf",
            str(tmp_path / "paper.pdf"),
            "--build-manifest",
            str(tmp_path / "build.json"),
            "--engine",
            "tectonic",
            "--tectonic-bin",
            str(tectonic),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "layout boxes" in result.stderr
    assert not (tmp_path / "paper.pdf").exists()


def test_tectonic_build_rejects_missing_page_count(tmp_path: Path):
    paper = _paper_tree(tmp_path)
    tectonic = _fake_tectonic(tmp_path, missing_page_count=True)
    result = subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--paper-dir",
            str(paper),
            "--output-pdf",
            str(tmp_path / "paper.pdf"),
            "--build-manifest",
            str(tmp_path / "build.json"),
            "--engine",
            "tectonic",
            "--tectonic-bin",
            str(tectonic),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "valid PDF page count" in result.stderr
    assert not (tmp_path / "paper.pdf").exists()


def test_pinned_installer_is_outside_repo_and_records_both_hashes():
    source = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert 'version="0.17.0"' in source
    assert 'asset="tectonic-${version}-x86_64-unknown-linux-gnu.tar.gz"' in source
    assert "1a715688baf591e650c8aeb160ae934e181685eecbb38b317de30b269ac5d606" in source
    assert "2b3a86250906c92ed0a3ae8aaa454ec55bd6cede8593b3e549640177f6aecaa3" in source
    assert "${XDG_CACHE_HOME:-$HOME/.cache}/long-context-position-bias" in source
    assert not os.path.commonpath(
        [str(ROOT), os.path.expanduser("~/.cache/long-context-position-bias")]
    ) == str(ROOT)
