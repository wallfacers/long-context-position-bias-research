from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_failure_cases.py"


def _row(
    sample_id: str,
    group_id: str,
    position: float,
    *,
    answer: str,
    answer_correct: bool,
    quote_correct: bool,
    valid_json: bool = True,
) -> dict:
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "task": "synthetic_test",
        "evaluation_mode": "test",
        "position_label": f"p{position}",
        "target_position": position,
        "actual_position": position,
        "answer_correct": answer_correct,
        "evidence_quotes_correct": quote_correct,
        "all_predicted_quotes_supported": quote_correct,
        "valid_json": valid_json,
        "target": {"answer": "DO-NOT-REDISTRIBUTE-TARGET"},
        "generated_text": "DO-NOT-REDISTRIBUTE-GENERATION",
        "parsed": {
            "answer": answer,
            "evidence_quote": "DO-NOT-REDISTRIBUTE-QUOTE",
        },
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_catalog_classifies_patterns_without_emitting_raw_text(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    _write(
        base,
        [
            _row("s0", "g", 0.0, answer="SECRET-A", answer_correct=True, quote_correct=False),
            _row("s1", "g", 0.5, answer="SECRET-B", answer_correct=False, quote_correct=True),
            _row("s2", "g", 1.0, answer="SECRET-A", answer_correct=True, quote_correct=False),
        ],
    )
    _write(
        treatment,
        [
            _row("s0", "g", 0.0, answer="SECRET-X", answer_correct=False, quote_correct=True),
            _row("s1", "g", 0.5, answer="SECRET-Y", answer_correct=True, quote_correct=True),
            _row("s2", "g", 1.0, answer="SECRET-X", answer_correct=False, quote_correct=True, valid_json=False),
        ],
    )
    output = tmp_path / "catalog"
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--run",
            f"base={base}",
            "--run",
            f"paired_evidence={treatment}",
            "--output-dir",
            str(output),
            "--max-examples",
            "2",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads((output / "failure_case_catalog.json").read_text())
    categories = payload["category_counts"]
    assert categories["edge_success_middle_failure"] == 1
    assert categories["middle_success_edge_failure"] == 1
    assert categories["base_only_answer_success"] == 2
    assert categories["treatment_only_answer_success"] == 1
    assert categories["treatment_quote_recovery"] == 2
    assert categories["answer_correct_quote_wrong"] == 2
    assert categories["answer_wrong_quote_correct"] >= 1
    assert categories["invalid_json"] == 1
    assert payload["status"] == "validated"
    assert payload["scope_denominators"] == {
        "row": 6,
        "group": 2,
        "cross_run": 3,
    }
    assert payload["category_scopes"]["answer_wrong"] == "row"
    assert payload["category_scopes"]["edge_success_middle_failure"] == "group"
    assert payload["category_scopes"]["treatment_quote_recovery"] == "cross_run"
    assert payload["category_rates"]["base_only_answer_success"] == 2 / 3
    assert len(payload["examples"]) <= 2 * len(categories)
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
    )
    for secret in (
        "DO-NOT-REDISTRIBUTE-TARGET",
        "DO-NOT-REDISTRIBUTE-GENERATION",
        "DO-NOT-REDISTRIBUTE-QUOTE",
        "SECRET-A",
        "SECRET-B",
        "SECRET-X",
        "SECRET-Y",
        str(tmp_path),
        '"sample_id":',
        '"group_id":',
        "generated_text_sha256",
        "parsed_answer_sha256",
    ):
        assert secret not in combined
    manifest = json.loads(
        (output / "failure_case_catalog.manifest.json").read_text()
    )
    markdown = (output / "failure_case_catalog.md").read_text(encoding="utf-8")
    csv_text = (output / "failure_case_catalog.csv").read_text(encoding="utf-8")
    assert "['" not in markdown
    assert "['" not in csv_text
    assert manifest["status"] == "validated"
    assert set(manifest["outputs"]) == {
        "failure_case_catalog.csv",
        "failure_case_catalog.json",
        "failure_case_catalog.md",
    }
    for name, record in manifest["outputs"].items():
        assert record["bytes"] == (output / name).stat().st_size
        assert record["sha256"] == hashlib.sha256((output / name).read_bytes()).hexdigest()


def test_catalog_rejects_duplicate_sample_ids(tmp_path: Path):
    source = tmp_path / "duplicate.jsonl"
    row = _row(
        "same", "g", 0.0, answer="SECRET", answer_correct=False, quote_correct=False
    )
    _write(source, [row, row])
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--run",
            f"base={source}",
            "--output-dir",
            str(tmp_path / "catalog"),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Duplicate sample_id" in result.stderr


def test_catalog_rejects_non_boolean_score(tmp_path: Path):
    source = tmp_path / "bad-score.jsonl"
    row = _row(
        "sample", "group", 0.0, answer="SECRET", answer_correct=False, quote_correct=False
    )
    row["answer_correct"] = "false"
    _write(source, [row])
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--run",
            f"base={source}",
            "--output-dir",
            str(tmp_path / "catalog"),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "answer_correct must be boolean" in result.stderr


def test_catalog_does_not_treat_inapplicable_quote_as_failure(tmp_path: Path):
    source = tmp_path / "inapplicable.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    row = _row(
        "sample", "group", 0.0, answer="SECRET", answer_correct=True, quote_correct=False
    )
    row["evidence_quotes_applicable"] = False
    row["all_predicted_quotes_supported_applicable"] = False
    _write(source, [row])
    _write(
        treatment,
        [
            _row(
                "sample",
                "group",
                0.0,
                answer="SECRET",
                answer_correct=True,
                quote_correct=True,
            )
        ],
    )
    output = tmp_path / "catalog"
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--run",
            f"base={source}",
            "--run",
            f"paired_evidence={treatment}",
            "--output-dir",
            str(output),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads((output / "failure_case_catalog.json").read_text())
    assert "answer_correct_quote_wrong" not in payload["category_counts"]
    assert payload["examples"] == []


def test_position_suite_runners_package_failure_catalogs():
    runners = (
        "run_autodl_nolima_gate.sh",
        "run_autodl_longbench_transfer.sh",
        "run_autodl_nolima_multiseed.sh",
        "run_autodl_longbench_multiseed.sh",
        "run_autodl_formal_eval.sh",
        "finalize_qwen_seed1_formal.sh",
    )
    for name in runners:
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "scripts/analyze_failure_cases.py" in text
        assert "failure_case_catalog.manifest.json" in text
        assert "--max-examples 5" in text
