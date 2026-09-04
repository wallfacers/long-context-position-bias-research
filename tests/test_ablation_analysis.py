from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_position_ablation.py"
SPEC = importlib.util.spec_from_file_location("analyze_position_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def rows_for_run(run_name: str, *, middle_correct: bool) -> list[dict[str, object]]:
    rows = []
    for group_index in range(2):
        for position in analysis.POSITIONS:
            correct = position != "p050" or middle_correct
            rows.append(
                {
                    "run_name": run_name,
                    "sample_id": f"sample-{group_index}@{position}",
                    "group_id": f"group-{group_index}",
                    "task": "kv",
                    "filler_type": "neutral",
                    "target_tokens": 8192,
                    "position_label": position,
                    "valid_json": True,
                    "answer_correct": correct,
                    "evidence_ids_correct": correct,
                    "evidence_quotes_correct": correct,
                    "all_predicted_quotes_supported": correct,
                }
            )
    return rows


class AblationAnalysisTests(unittest.TestCase):
    def test_group_paired_summary_captures_middle_penalty(self) -> None:
        rows_by_run = {
            run_name: rows_for_run(
                run_name, middle_correct=run_name != "base"
            )
            for run_name in analysis.RUN_ORDER
        }
        grouped, condition_groups = analysis.build_group_tables(rows_by_run)
        base, profiles = analysis.summarize_selection(
            grouped["base"], condition_groups
        )
        trained, _ = analysis.summarize_selection(
            grouped["paired_evidence"], condition_groups
        )
        self.assertEqual(base["mean_worst_position_accuracy"], 0.0)
        self.assertEqual(base["mean_position_gap"], 1.0)
        self.assertEqual(base["mean_middle_penalty"], 1.0)
        self.assertEqual(trained["mean_worst_position_accuracy"], 1.0)
        self.assertEqual(trained["mean_position_gap"], 0.0)
        self.assertEqual(profiles[("kv", 8192, "p050")], 0.0)

    def test_factorial_interaction_weight_signs(self) -> None:
        weights = analysis.CONTRAST_WEIGHTS["pairing_x_supervision_interaction"]
        summaries = {
            run_name: {"answer_correct": value}
            for run_name, value in {
                "base": 0.0,
                "paired_evidence": 0.9,
                "paired_answer": 0.6,
                "independent_evidence": 0.7,
                "independent_answer": 0.5,
            }.items()
        }
        self.assertAlmostEqual(
            analysis.weighted_contrast(summaries, weights, "answer_correct"),
            0.1,
        )

    def test_holm_adjustment_is_monotone_and_bounded(self) -> None:
        adjusted = analysis.holm_adjust([("a", 0.01), ("b", 0.03), ("c", 0.5)])
        self.assertEqual(adjusted["a"], 0.03)
        self.assertEqual(adjusted["b"], 0.06)
        self.assertEqual(adjusted["c"], 0.5)
        self.assertTrue(all(0 <= value <= 1 for value in adjusted.values()))

    def test_cli_writes_reproducible_analysis_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "analysis"
            results.mkdir()
            for run_name in analysis.RUN_ORDER:
                rows = rows_for_run(run_name, middle_correct=run_name != "base")
                (results / f"{run_name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--results-dir",
                    str(results),
                    "--output-dir",
                    str(output),
                    "--bootstrap-replicates",
                    "100",
                    "--seed",
                    "7",
                ]
                self.assertEqual(analysis.main(), 0)
            finally:
                sys.argv = old_argv
            report = json.loads(
                (output / "ablation_analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["bootstrap"]["replicates"], 100)
            self.assertEqual(report["rows_per_run"]["base"], 14)
            self.assertEqual(
                report["generation_diagnostics"]["base"]["finish_reason_counts"],
                {"unknown": 14},
            )
            first_screen = report["exploratory_screening"]["results"][0]
            self.assertIn(
                "edge_accuracy_drop_no_more_than_2pp", first_screen["checks"]
            )
            self.assertEqual(
                report["exploratory_screening"]["criteria"][
                    "edge_accuracy_delta_floor"
                ],
                -0.02,
            )
            self.assertTrue((output / "paired_bootstrap_indices.jsonl.gz").is_file())
            self.assertTrue((output / "ablation_contrasts.csv").is_file())


if __name__ == "__main__":
    unittest.main()
