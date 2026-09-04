import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_nolima_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("prepare_nolima_diagnostics", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

sys.path.insert(0, str(ROOT / "src"))
from position_bias_research.tokenization import WhitespaceTokenCounter


class NoLiMaDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "data/ood_nolima/hard_gate.jsonl"
        if not path.is_file():
            raise unittest.SkipTest(
                "NoLiMa benchmark payload is license-excluded; reconstruct it from the frozen manifest"
            )
        with path.open(encoding="utf-8") as handle:
            first = json.loads(next(handle))
            cls.rows = [first]
            for line in handle:
                row = json.loads(line)
                if row["group_id"] == first["group_id"]:
                    cls.rows.append(row)
        assert len(cls.rows) == 7

    def test_removing_needle_recovers_one_shared_book(self):
        recovered = set()
        for row in self.rows:
            book, _ = module.split_prompt(row["prompt"])
            recovered.add(
                module.remove_exact_needle(
                    book, row["target"]["evidence_quotes"][0]
                )
            )
        self.assertEqual(len(recovered), 1)

    def test_oracles_are_deduplicated_across_positions(self):
        derived = module.derive_rows(self.rows, WhitespaceTokenCounter())
        counts = {
            mode: sum(row["evaluation_mode"] == mode for row in derived)
            for mode in module.MODES
        }
        self.assertEqual(
            counts,
            {"locate_only": 7, "oracle_long": 1, "oracle_short": 1},
        )
        for row in derived:
            quote = row["target"]["evidence_quotes"][0]
            self.assertEqual(row["prompt"].count(quote), 1)
            self.assertGreater(row["actual_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
