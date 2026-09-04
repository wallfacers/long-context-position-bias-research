import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_seed_level_factorial_results.py"
SPEC = importlib.util.spec_from_file_location("plot_seed_level_factorial_results", SCRIPT)
plots = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plots)


class SeedLevelFactorialPlotTest(unittest.TestCase):
    def test_writes_publication_formats_table_alt_text_and_manifest(self):
        rows = []
        for family_index, family in enumerate(("Mistral-7B-v0.3", "Qwen2.5-7B")):
            for run_index, run_name in enumerate(plots.RUN_STYLES):
                for position_index, position in enumerate(plots.POSITIONS):
                    value = min(
                        0.98,
                        0.35
                        + 0.08 * family_index
                        + 0.04 * run_index
                        + 0.01 * position_index,
                    )
                    rows.append(
                        {
                            "family": family,
                            "run_name": run_name,
                            "task": "nolima_hard",
                            "target_tokens": 8192,
                            "position_label": position,
                            "n_seeds": 1 if run_name == "base" else 2,
                            "mean": value,
                            "sd": 0.01,
                            "ci95_low": value - 0.05,
                            "ci95_high": value + 0.05,
                            "min": value - 0.01,
                            "max": value + 0.01,
                            "seeds": [] if run_name == "base" else [20260826, 20260827],
                            "seed_estimates": (
                                [value]
                                if run_name == "base"
                                else [value - 0.01, value + 0.01]
                            ),
                        }
                    )
        report = {
            "schema_version": "seed-level-analysis-v1",
            "analysis_kind": "factorial",
            "primary_training_seed_summary": True,
            "confirmatory_only_primary_summary": False,
            "primary_statuses_by_family": {
                "Qwen2.5-7B": ["corrective"],
                "Mistral-7B-v0.3": ["confirmatory"],
            },
            "position_profiles": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "analysis.json"
            output = root / "figures"
            analysis.write_text(json.dumps(report), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--analysis",
                    str(analysis),
                    "--output-dir",
                    str(output),
                    "--basename",
                    "paper_curve",
                ]
                self.assertEqual(plots.main(), 0)
            finally:
                sys.argv = old_argv
            for suffix in ("pdf", "svg", "png", "csv", "alt.txt", "manifest.json"):
                self.assertTrue((output / f"paper_curve.{suffix}").is_file())
            manifest = json.loads(
                (output / "paper_curve.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], "seed-level-factorial-figure-v2")
            self.assertEqual(manifest["status"], "validated")
            self.assertFalse(manifest["confirmatory_only"])
            self.assertTrue(manifest["corrective_plus_confirmatory_primary"])
            self.assertEqual(manifest["task_case_weights"], {"nolima_hard": 1})
            self.assertIn(
                "exact", (output / "paper_curve.alt.txt").read_text().lower()
            )
            self.assertIn(
                "corrective", (output / "paper_curve.alt.txt").read_text().lower()
            )

    def test_rejects_confirmatory_only_relabeling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "analysis.json"
            analysis.write_text(
                json.dumps(
                    {
                        "schema_version": "seed-level-analysis-v1",
                        "analysis_kind": "factorial",
                        "primary_training_seed_summary": True,
                        "confirmatory_only_primary_summary": True,
                        "primary_statuses_by_family": plots.EXPECTED_PRIMARY_STATUSES,
                        "position_profiles": [],
                    }
                ),
                encoding="utf-8",
            )
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--analysis",
                    str(analysis),
                    "--output-dir",
                    str(root / "figures"),
                ]
                with self.assertRaisesRegex(SystemExit, "Qwen-corrective"):
                    plots.main()
            finally:
                sys.argv = old_argv

    def test_case_weighted_aggregation_uses_semantic_case_counts(self):
        rows = []
        task_values = {
            "nolima_onehop": [0.0, 0.2],
            "nolima_twohop": [0.5, 0.7],
            "nolima_twohop2": [1.0, 0.8],
        }
        for task, values in task_values.items():
            rows.append(
                {
                    "family": "Qwen2.5-7B",
                    "run_name": "independent_answer",
                    "task": task,
                    "target_tokens": 8192,
                    "position_label": "p050",
                    "seeds": [20260826, 20260827],
                    "seed_estimates": values,
                }
            )
        aggregated, weights = plots.aggregate_tasks(rows)
        self.assertEqual(weights, plots.NOLIMA_CASE_WEIGHTS)
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["task"], "case_weighted_nolima_hard")
        self.assertAlmostEqual(aggregated[0]["seed_estimates"][0], 0.5)
        self.assertAlmostEqual(aggregated[0]["seed_estimates"][1], 0.62)
        self.assertAlmostEqual(aggregated[0]["mean"], 0.56)


if __name__ == "__main__":
    unittest.main()
