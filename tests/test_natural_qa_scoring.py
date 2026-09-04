from __future__ import annotations

import ast
import json
import re
import string
import types
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts" / "evaluate_vllm.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
NAMES = {"normalize_english_answer", "token_f1", "answer_score", "score"}
SELECTED = [
    node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in NAMES
]
MODULE = types.ModuleType("natural_scoring_under_test")
MODULE.__dict__.update(
    {"re": re, "string": string, "Counter": Counter, "Any": Any, "json": json}
)
exec(compile(ast.Module(body=SELECTED, type_ignores=[]), str(ROOT), "exec"), MODULE.__dict__)


def natural_row() -> dict:
    return {
        "prompt": "Passage text",
        "target": {
            "answer": "The Miller case",
            "answers": ["Miller case", "the Miller case"],
            "evidence_ids": [],
            "evidence_quotes": [],
        },
        "metadata": {
            "answer_metric": "qa_f1_en",
            "evidence_id_applicable": False,
            "evidence_quote_applicable": False,
        },
    }


class NaturalQAScoringTest(unittest.TestCase):
    def test_qa_f1_uses_best_reference_and_marks_evidence_na(self) -> None:
        result = MODULE.score(
            natural_row(),
            json.dumps(
                {
                    "answer": "Miller case",
                    "evidence_ids": [],
                    "evidence_quotes": [],
                    "confidence": 0.9,
                }
            ),
        )
        self.assertEqual(result["answer_score"], 1.0)
        self.assertTrue(result["answer_correct"])
        self.assertIsNone(result["evidence_ids_correct"])
        self.assertIsNone(result["evidence_quotes_correct"])

    def test_partial_answer_has_fractional_f1(self) -> None:
        row = natural_row()
        row["target"]["answers"] = ["New York City"]
        value = MODULE.score(
            row,
            json.dumps(
                {
                    "answer": "New York",
                    "evidence_ids": [],
                    "evidence_quotes": [],
                    "confidence": 0.5,
                }
            ),
        )
        self.assertAlmostEqual(value["answer_score"], 0.8)
        self.assertFalse(value["answer_correct"])


if __name__ == "__main__":
    unittest.main()
