import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_seed_level_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_seed_level_results", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class SeedLevelResultsTest(unittest.TestCase):
    def test_corrective_status_is_explicitly_supported(self):
        family, seed, status, path = module.parse_analysis(
            "Qwen:20260825:corrective:result.json"
        )
        self.assertEqual((family, seed, status), ("Qwen", 20260825, "corrective"))
        self.assertEqual(path, Path("result.json"))
        self.assertIn("corrective", module.PRIMARY_STATUSES)

    def test_two_seed_interval_uses_seed_as_unit(self):
        summary = module.mean_interval([0.1, 0.3])
        self.assertEqual(summary["n_seeds"], 2)
        self.assertAlmostEqual(summary["mean"], 0.2)
        self.assertGreater(summary["ci95_high"], 1.0)
        self.assertLess(summary["ci95_low"], -0.5)

    def test_fixed_base_is_deduplicated_not_counted_as_training_seeds(self):
        summary = module.fixed_base_interval([0.7, 0.7, 0.7])
        self.assertEqual(summary["n_seeds"], 1)
        self.assertIsNone(summary["ci95_low"])
        self.assertTrue(summary["fixed_untrained_base"])
        self.assertEqual(summary["reused_analysis_copies"], 3)

    def test_fixed_base_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            module.fixed_base_interval([0.7, 0.71])

    def test_extracts_factorial_run_and_contrast_estimates(self):
        report = {
            "schema_version": "matched-factorial-analysis-v1",
            "run_summary_intervals": {
                "base": {"answer_correct": {"estimate": 0.5}}
            },
            "contrasts": {
                "paired_minus_independent_main_effect": {
                    "statistics": {"answer_correct": {"estimate": 0.1}}
                }
            },
        }
        kind, values = module.extract(report)
        self.assertEqual(kind, "factorial")
        self.assertEqual(values[("run:base", "answer_correct")], 0.5)
        self.assertEqual(
            values[("contrast:paired_minus_independent_main_effect", "answer_correct")],
            0.1,
        )

    def test_extracts_unique_position_profile_cells(self):
        report = {
            "schema_version": "matched-factorial-analysis-v1",
            "position_profiles": [
                {
                    "run_name": "paired_answer",
                    "task": "nolima_hard",
                    "target_tokens": 8192,
                    "position_label": "p050",
                    "answer_accuracy": 0.75,
                }
            ],
        }
        values = module.extract_position_profiles(report)
        self.assertEqual(
            values[("paired_answer", "nolima_hard", 8192, "p050")], 0.75
        )

    def test_family_specific_audited_length_grids_are_allowed(self):
        def report(length):
            return {
                "schema_version": "matched-factorial-analysis-v1",
                "run_summary_intervals": {
                    "base": {"answer_correct": {"estimate": 0.5}}
                },
                "contrasts": {},
                "position_profiles": [
                    {
                        "run_name": "base",
                        "task": "rule",
                        "target_tokens": length,
                        "position_label": "p050",
                        "answer_accuracy": 0.5,
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = []
            for family, length in (("Qwen", 32768), ("Mistral", 32512)):
                for seed in (1, 2):
                    path = root / f"{family}-{seed}.json"
                    path.write_text(json.dumps(report(length)), encoding="utf-8")
                    arguments.extend(
                        ["--analysis", f"{family}:{seed}:confirmatory:{path}"]
                    )
            output = root / "output"
            old_argv = sys.argv
            try:
                sys.argv = [str(SCRIPT), *arguments, "--output-dir", str(output)]
                self.assertEqual(module.main(), 0)
            finally:
                sys.argv = old_argv
            payload = json.loads(
                (output / "seed_level_analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {row["target_tokens"] for row in payload["position_profiles"]},
                {32512, 32768},
            )


if __name__ == "__main__":
    unittest.main()
