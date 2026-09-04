from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AggregateResultsTest(unittest.TestCase):
    def test_non_applicable_evidence_ids_are_reported_as_null(self) -> None:
        row = {
            "valid_json": True,
            "answer_correct": True,
            "evidence_ids_correct": None,
            "evidence_ids_applicable": False,
            "evidence_quotes_correct": True,
            "all_predicted_quotes_supported": True,
        }
        report = __import__("scripts.aggregate_results", fromlist=["rates"]).rates([row])
        self.assertIsNone(report["evidence_ids_correct"])
        self.assertEqual(report["evidence_ids_correct_n"], 0)
        self.assertEqual(report["answer_correct_n"], 1)

    def test_diagnostic_modes_are_not_mixed_in_mode_summary(self) -> None:
        base = {
            "run_name": "base",
            "task": "kv",
            "filler_type": "neutral",
            "target_tokens": 8192,
            "position_label": "p050",
            "valid_json": True,
            "answer_correct": True,
            "evidence_ids_correct": True,
            "evidence_quotes_correct": True,
            "all_predicted_quotes_supported": True,
            "parsed": {"answer": "VALUE"},
        }
        rows = [
            base
            | {
                "sample_id": "free-sample",
                "group_id": "free-group",
                "evaluation_mode": "free",
            },
            base
            | {
                "sample_id": "locate-sample",
                "group_id": "locate-group",
                "evaluation_mode": "locate_only",
                "answer_correct": False,
                "parsed": None,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.jsonl"
            output = root / "summary.json"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "aggregate_results.py"),
                    str(source),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["overall_by_run"][0]["n"], 2)
        by_mode = {
            row["evaluation_mode"]: row for row in report["overall_by_run_mode"]
        }
        self.assertEqual(set(by_mode), {"free", "locate_only"})
        self.assertEqual(by_mode["free"]["answer_correct"], 1.0)
        self.assertEqual(by_mode["locate_only"]["answer_correct"], 0.0)
        consistency = {
            row["evaluation_mode"]: row for row in report["group_consistency"]
        }
        self.assertEqual(set(consistency), {"free", "locate_only"})
        self.assertEqual(consistency["free"]["same_answer_across_positions"], 1.0)
        self.assertEqual(
            consistency["locate_only"]["same_answer_across_positions"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
