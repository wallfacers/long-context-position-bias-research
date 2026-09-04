import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_nolima_mechanisms.py"
SPEC = importlib.util.spec_from_file_location("analyze_nolima_mechanisms", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class NoLiMaMechanismAnalysisTest(unittest.TestCase):
    def test_derived_bottleneck_metrics(self):
        raw = {
            "free_answer": 0.4,
            "free_quote": 0.3,
            "free_worst_answer": 0.1,
            "locate_quote": 0.7,
            "locate_worst_quote": 0.5,
            "locate_supported": 0.9,
            "oracle_long_answer": 0.8,
            "oracle_short_answer": 0.95,
        }
        values = module.add_derived(raw)
        self.assertAlmostEqual(values["retrieval_recovery"], 0.4)
        self.assertAlmostEqual(values["localization_gain"], 0.4)
        self.assertAlmostEqual(values["long_distraction_penalty"], 0.15)
        self.assertAlmostEqual(values["residual_reasoning_error"], 0.05)

    def test_case_selection_preserves_paired_run_structure(self):
        records = {}
        for run_index, run in enumerate(module.RUN_ORDER):
            records[run] = []
            for case_index, case in enumerate(("case-a", "case-b")):
                record = {
                    "case_id": case,
                    **{
                        metric: (run_index + case_index) / 10
                        for metric in module.RAW_METRICS
                    },
                }
                records[run].append(record)
        values = module.summarize(records, ["case-a", "case-a"])
        self.assertEqual(values["base"]["free_answer"], 0.0)
        self.assertEqual(values["independent_answer"]["free_answer"], 0.1)
        self.assertAlmostEqual(
            values["base"]["residual_reasoning_error"], 1.0
        )

    def test_case_strata_preserve_task_composition(self):
        records = [
            {"case_id": "case-a", "task": "onehop"},
            {"case_id": "case-a", "task": "onehop"},
            {"case_id": "case-b", "task": "twohop"},
            {"case_id": "case-c", "task": "twohop"},
        ]
        self.assertEqual(
            module.case_strata(records),
            {"onehop": ["case-a"], "twohop": ["case-b", "case-c"]},
        )


if __name__ == "__main__":
    unittest.main()
