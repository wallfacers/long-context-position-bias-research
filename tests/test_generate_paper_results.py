from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_paper_results.py"
SPEC = importlib.util.spec_from_file_location("generate_paper_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GeneratePaperResultsTest(unittest.TestCase):
    def family(self):
        payload = {}
        for index, run in enumerate(MODULE.TABLE_RUNS):
            for statistic, value in {
                "answer_correct": 0.8 + index / 100,
                "mean_worst_answer_accuracy": 0.7 + index / 100,
                "mean_answer_position_gap": 0.2 - index / 100,
                "evidence_quotes_correct": 0.6 + index / 100,
            }.items():
                payload[f"run:{run}|{statistic}"] = {"mean": value}
        return payload

    def test_rows_cover_all_factorial_cells_and_escape_family(self):
        rows = MODULE.build_rows({"families": {"Qwen_7B": self.family()}})
        self.assertEqual(len(rows), 7)
        self.assertIn(r"Qwen\_7B", rows[0])
        self.assertIn("-- & --", rows[0])
        self.assertIn("Independent & Answer", rows[1])
        self.assertIn("86.0", rows[-1])

    def test_rows_render_non_applicable_quote_metric_as_na_not_zero(self):
        family = self.family()
        del family["run:independent_answer|evidence_quotes_correct"]
        rows = MODULE.build_rows({"families": {"Qwen": family}})
        independent_answer = rows[1].split(" & ")
        self.assertEqual(independent_answer[-1], r"-- \\")
        self.assertNotIn("0.0", rows[1])

    def test_longbench_rows_cover_base_and_six_treatments(self):
        family = {}
        for index, run in enumerate(MODULE.TABLE_RUNS):
            for slice_name in (
                "longbench_hotpotqa",
                "longbench_2wikimqa",
                "longbench_musique",
                "overall",
            ):
                family[f"run:{run}|{slice_name}"] = {"mean": 0.4 + index / 100}
        rows = MODULE.build_longbench_rows({"families": {"Qwen": family}})
        self.assertEqual(len(rows), 7)
        self.assertIn("40.0", rows[0])
        self.assertIn("46.0", rows[-1])

    def test_factorial_contrast_rows_report_seed_level_intervals(self):
        family = {}
        for contrast_index, (contrast, _) in enumerate(
            MODULE.KEY_FACTORIAL_CONTRASTS
        ):
            for statistic, value in (
                ("mean_worst_answer_accuracy", 0.01 + contrast_index / 100),
                ("mean_answer_position_gap", -0.02 - contrast_index / 100),
            ):
                family[f"contrast:{contrast}|{statistic}"] = {
                    "mean": value,
                    "ci95_low": value - 0.01,
                    "ci95_high": value + 0.01,
                }
        report = {"families": {"Qwen_7B": family}}
        longbench_family = {
            f"contrast:{contrast}|overall": {
                "mean": 0.005,
                "ci95_low": -0.005,
                "ci95_high": 0.015,
            }
            for contrast, _ in MODULE.KEY_FACTORIAL_CONTRASTS
        }
        rows = MODULE.build_factorial_contrast_rows(
            report, report, {"families": {"Qwen_7B": longbench_family}}
        )
        self.assertEqual(len(rows), 4)
        self.assertIn(r"Qwen\_7B & Paired $-$ independent", rows[0])
        self.assertIn("+1.0 [+0.0, +2.0]", rows[0])
        self.assertIn("-2.0 [-3.0, -1.0]", rows[0])
        self.assertIn("+0.5 [-0.5, +1.5]", rows[0])

    def test_exploratory_rule_macros_preserve_position_failure(self):
        intervals = {
            run: {
                "answer_correct": {"estimate": 0.97 + index / 200},
                "mean_worst_answer_accuracy": {"estimate": 0.80 + index / 100},
                "mean_answer_position_gap": {"estimate": 0.20 - index / 100},
            }
            for index, run in enumerate(MODULE.VARIANTS)
        }
        metrics = MODULE.exploratory_rule_metrics(
            {"run_summary_intervals": intervals}
        )
        self.assertAlmostEqual(metrics["ExploratoryRuleAnswerMin"], 0.97)
        self.assertAlmostEqual(metrics["ExploratoryRuleAnswerMax"], 0.995)
        self.assertAlmostEqual(metrics["ExploratoryPairedAnswerWorst"], 0.83)
        self.assertAlmostEqual(metrics["ExploratoryIndependentEvidenceWorst"], 0.82)

    def test_regression_rows_cover_both_capability_checks_and_ni(self):
        mmlu = {
            "run_intervals": {
                run: {"estimate": 0.50 + index / 100}
                for index, run in enumerate(MODULE.TABLE_RUNS)
            },
            "noninferiority_to_base": {
                run: {"passes_if_ci95_low_above_margin": index % 2 == 0}
                for index, run in enumerate(MODULE.VARIANTS)
            },
        }
        ifeval = {
            "run_intervals": {
                run: {"strict_prompt": {"estimate": 0.60 + index / 100}}
                for index, run in enumerate(MODULE.TABLE_RUNS)
            },
            "noninferiority_to_base_strict_prompt": {
                run: {"passes_if_ci95_low_above_margin": index % 2 == 1}
                for index, run in enumerate(MODULE.VARIANTS)
            },
        }
        rows = MODULE.build_regression_rows([("Qwen_7B", mmlu, ifeval)])
        self.assertEqual(len(rows), 7)
        self.assertIn(r"Qwen\_7B & -- & -- & 50.0 & -- & 60.0 & --", rows[0])
        self.assertIn("Independent & Answer & 51.0 & Pass & 61.0 & Fail", rows[1])
        self.assertIn("Paired & Exact evidence & 56.0 & Fail & 66.0 & Pass", rows[-1])

    def test_mechanism_rows_use_the_frozen_four_corner_subset(self):
        report = {
            "run_intervals": {
                run: {
                    metric: {"estimate": 0.40 + run_index / 100 + metric_index / 1000}
                    for metric_index, metric in enumerate(
                        (
                            "free_answer",
                            "locate_quote",
                            "oracle_long_answer",
                            "oracle_short_answer",
                        )
                    )
                }
                for run_index, run in enumerate(
                    (
                        "base",
                        "independent_answer",
                        "independent_evidence",
                        "paired_answer",
                        "paired_evidence",
                    )
                )
            }
        }
        rows = MODULE.build_mechanism_rows([("Mistral", report)])
        self.assertEqual(len(rows), 5)
        self.assertIn("Mistral & -- & -- & 40.0 & 40.1 & 40.2 & 40.3", rows[0])
        self.assertIn("Paired & Exact evidence & 44.0 & 44.1 & 44.2 & 44.3", rows[-1])

    def test_cli_hashes_all_factorial_regression_and_mechanism_sources(self):
        longbench_family = {}
        for index, run in enumerate(MODULE.TABLE_RUNS):
            for slice_name in (
                "longbench_hotpotqa",
                "longbench_2wikimqa",
                "longbench_musique",
                "overall",
            ):
                longbench_family[f"run:{run}|{slice_name}"] = {"mean": 0.4 + index / 100}
        general = {
            "schema_version": "general-regression-analysis-v1",
            "run_intervals": {
                run: {"estimate": 0.5} for run in MODULE.TABLE_RUNS
            },
            "noninferiority_to_base": {
                run: {"passes_if_ci95_low_above_margin": True}
                for run in MODULE.VARIANTS
            },
        }
        ifeval = {
            "schema_version": "ifeval-regression-analysis-v1",
            "run_intervals": {
                run: {"strict_prompt": {"estimate": 0.6}}
                for run in MODULE.TABLE_RUNS
            },
            "noninferiority_to_base_strict_prompt": {
                run: {"passes_if_ci95_low_above_margin": True}
                for run in MODULE.VARIANTS
            },
        }
        mechanism_runs = (
            "base",
            "independent_answer",
            "independent_evidence",
            "paired_answer",
            "paired_evidence",
        )
        mechanism = {
            "schema_version": "nolima-mechanism-analysis-v1",
            "run_intervals": {
                run: {
                    metric: {"estimate": 0.7}
                    for metric in (
                        "free_answer",
                        "locate_quote",
                        "oracle_long_answer",
                        "oracle_short_answer",
                    )
                }
                for run in mechanism_runs
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, payload):
                path = root / name
                path.write_text(json.dumps(payload), encoding="utf-8")
                return path

            factorial_family = self.family()
            for contrast, _ in MODULE.KEY_FACTORIAL_CONTRASTS:
                for statistic in (
                    "mean_worst_answer_accuracy",
                    "mean_answer_position_gap",
                ):
                    factorial_family[f"contrast:{contrast}|{statistic}"] = {
                        "mean": 0.01,
                        "ci95_low": -0.01,
                        "ci95_high": 0.03,
                    }
            common = {
                "schema_version": "seed-level-analysis-v1",
                "primary_training_seed_summary": True,
                "confirmatory_only_primary_summary": False,
                "primary_statuses_by_family": {
                    "Qwen2.5-7B": ["corrective"],
                    "Mistral-7B-v0.3": ["confirmatory"],
                },
                "families": {
                    "Qwen2.5-7B": factorial_family,
                    "Mistral-7B-v0.3": dict(factorial_family),
                },
            }
            rule = write("rule.json", {**common, "analysis_kind": "factorial"})
            nolima = write("nolima.json", {**common, "analysis_kind": "factorial"})
            for contrast, _ in MODULE.KEY_FACTORIAL_CONTRASTS:
                longbench_family[f"contrast:{contrast}|overall"] = {
                    "mean": 0.005,
                    "ci95_low": -0.005,
                    "ci95_high": 0.015,
                }
            longbench = write(
                "longbench.json",
                {
                    **common,
                    "analysis_kind": "natural_transfer",
                    "families": {
                        "Qwen2.5-7B": longbench_family,
                        "Mistral-7B-v0.3": dict(longbench_family),
                    },
                },
            )
            exploratory_rule = write(
                "exploratory-rule.json",
                {
                    "schema_version": "matched-factorial-analysis-v1",
                    "run_summary_intervals": {
                        run: {
                            "answer_correct": {"estimate": 0.9},
                            "mean_worst_answer_accuracy": {"estimate": 0.8},
                            "mean_answer_position_gap": {"estimate": 0.1},
                        }
                        for run in MODULE.TABLE_RUNS
                    },
                },
            )
            qwen_mmlu = write("qwen-mmlu.json", general)
            qwen_ifeval = write("qwen-ifeval.json", ifeval)
            qwen_mechanism = write("qwen-mechanism.json", mechanism)
            mistral_mmlu = write("mistral-mmlu.json", general)
            mistral_ifeval = write("mistral-ifeval.json", ifeval)
            mistral_mechanism = write("mistral-mechanism.json", mechanism)
            output_tex, output_manifest = root / "results.tex", root / "manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--rule",
                    str(rule),
                    "--nolima",
                    str(nolima),
                    "--longbench",
                    str(longbench),
                    "--qwen-exploratory-rule",
                    str(exploratory_rule),
                    "--qwen-mmlu",
                    str(qwen_mmlu),
                    "--qwen-ifeval",
                    str(qwen_ifeval),
                    "--qwen-mechanisms",
                    str(qwen_mechanism),
                    "--mistral-mmlu",
                    str(mistral_mmlu),
                    "--mistral-ifeval",
                    str(mistral_ifeval),
                    "--mistral-mechanisms",
                    str(mistral_mechanism),
                    "--output-tex",
                    str(output_tex),
                    "--output-manifest",
                    str(output_manifest),
                ],
                check=True,
            )
            self.assertIn(r"\newcommand{\GeneratedRegressionRows}", output_tex.read_text())
            self.assertIn(r"\newcommand{\GeneratedMechanismRows}", output_tex.read_text())
            self.assertNotIn("RuleAnswerBest", output_tex.read_text())
            self.assertNotIn("NoLiMaWorstBest", output_tex.read_text())
            self.assertNotIn("LongBenchBest", output_tex.read_text())
            self.assertIn(
                r"\newcommand{\GeneratedFactorialContrastRows}",
                output_tex.read_text(),
            )
            manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["sources"]), 10)
            self.assertFalse(manifest["confirmatory_only"])
            self.assertFalse(manifest["primary_summaries_confirmatory_only"])
            self.assertTrue(manifest["corrective_plus_confirmatory_primary"])
            self.assertEqual(
                manifest["primary_statuses_by_family"],
                {
                    "Qwen2.5-7B": ["corrective"],
                    "Mistral-7B-v0.3": ["confirmatory"],
                },
            )
            self.assertEqual(
                manifest["labeled_exploratory_sources"],
                ["qwen_exploratory_rule"],
            )

    def test_primary_designations_reject_confirmatory_only_relabeling(self):
        report = {
            "primary_statuses_by_family": MODULE.EXPECTED_PRIMARY_STATUSES,
            "confirmatory_only_primary_summary": True,
        }
        with self.assertRaisesRegex(ValueError, "cannot be labeled confirmatory-only"):
            MODULE.validate_primary_designations((report, report, report))


if __name__ == "__main__":
    unittest.main()
