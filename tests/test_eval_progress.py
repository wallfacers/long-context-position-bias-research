from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "estimate_eval_progress.py"
SPEC = importlib.util.spec_from_file_location("estimate_eval_progress", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvalProgressTest(unittest.TestCase):
    def test_last_measured_rate_and_jsonl_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "eval.log"
            log.write_text(
                "run=a saved=4/10 sec/sample=2.500\n"
                "run=a saved=8/10 sec/sample=2.125\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.last_rate(log)["seconds_per_sample"], 2.125)
            result = root / "a.jsonl"
            result.write_text("{}\n{}\n", encoding="utf-8")
            self.assertEqual(MODULE.count_jsonl(result), 2)


if __name__ == "__main__":
    unittest.main()
