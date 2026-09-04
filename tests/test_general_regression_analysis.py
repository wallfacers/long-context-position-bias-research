from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_general_regression.py"
SPEC = importlib.util.spec_from_file_location("analyze_general_regression", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class GeneralRegressionAnalysisTest(unittest.TestCase):
    def test_noninferiority_and_paired_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_args = []
            for run_index, run in enumerate(analysis.RUN_ORDER):
                path = root / f"{run}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for subject in ("abstract_algebra", "anatomy"):
                        for index in range(10):
                            is_correct = index < 5 + min(run_index, 2)
                            handle.write(
                                json.dumps(
                                    {
                                        "sample_id": f"{subject}/{index}",
                                        "task": f"mmlu_{subject}",
                                        "answer_score": int(is_correct),
                                        "target": {"answer": "A" if is_correct else "B"},
                                        "generated_text": '{"answer": "A"}',
                                        "parsed": {"answer": "A"},
                                        "valid_json": True,
                                        "finish_reason": "stop",
                                    }
                                )
                                + "\n"
                            )
                run_args.extend(["--run", f"{run}={path}"])
            output = root / "analysis"
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    *run_args,
                    "--output-dir",
                    str(output),
                    "--bootstrap-replicates",
                    "100",
                ]
                self.assertEqual(analysis.main(), 0)
            finally:
                sys.argv = old_argv
            report = json.loads((output / "general_regression_analysis.json").read_text())
            self.assertEqual(report["rows_per_run"]["base"], 20)
            self.assertTrue(
                report["noninferiority_to_base"]["paired_evidence"]["passes_if_ci95_low_above_margin"]
            )
            self.assertEqual(report["bootstrap"]["replicates"], 100)
            self.assertEqual(
                report["scoring_protocol"]["name"], "format-robust-option-extraction-v1"
            )

    def test_truncated_json_option_is_scored_without_valid_json(self) -> None:
        row = {
            "target": {"answer": "C"},
            "generated_text": '{"answer": "C", "confidence": 0.9',
            "parsed": None,
            "valid_json": False,
        }
        self.assertEqual(
            analysis.extract_option(row), ("C", "truncated_or_embedded_json")
        )
        self.assertEqual(analysis.target_option(row), "C")

    def test_ambiguous_answer_phrases_are_not_guessed(self) -> None:
        row = {
            "generated_text": "The answer is A, although option B was also considered.",
            "parsed": None,
        }
        self.assertEqual(analysis.extract_option(row), (None, "unextractable"))


if __name__ == "__main__":
    unittest.main()
