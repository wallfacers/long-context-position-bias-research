from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_factorial_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_factorial_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def rows_for_run(
    run_name: str,
    answer_value: bool,
    evidence_ids_applicable: bool = True,
    evidence_quotes_applicable: bool = True,
):
    rows = []
    for group_index in range(3):
        for position in analysis.POSITIONS:
            rows.append(
                {
                    "run_name": run_name,
                    "sample_id": f"sample-{group_index}@{position}",
                    "group_id": f"group-{group_index}",
                    "evaluation_mode": "free",
                    "task": "nolima_twohop",
                    "filler_type": "nolima_book",
                    "target_tokens": 8192,
                    "position_label": position,
                    "valid_json": True,
                    "answer_correct": answer_value or position != "p050",
                    "evidence_ids_applicable": evidence_ids_applicable,
                    "evidence_ids_correct": True if evidence_ids_applicable else None,
                    "evidence_quotes_applicable": evidence_quotes_applicable,
                    "evidence_quotes_correct": (
                        answer_value if evidence_quotes_applicable else None
                    ),
                    "all_predicted_quotes_supported_applicable": evidence_quotes_applicable,
                    "all_predicted_quotes_supported": (
                        answer_value if evidence_quotes_applicable else None
                    ),
                    "parsed": {"answer": "A" if answer_value else position},
                    "finish_reason": "stop",
                    "output_tokens": 20,
                    "metadata": {
                        "benchmark": "NoLiMa",
                        "case_id": f"case-{group_index % 2}",
                    },
                }
            )
    return rows


class FactorialAnalysisTest(unittest.TestCase):
    def test_generation_results_preserve_safe_input_metadata(self) -> None:
        source = (ROOT / "scripts/evaluate_suite_vllm.py").read_text(encoding="utf-8")
        self.assertIn('"metadata": row.get("metadata", {})', source)

    def test_factorial_contrast_weights(self) -> None:
        weights = analysis.CONTRASTS["pairing_x_evidence_vs_answer"]
        self.assertEqual(weights["paired_evidence"], 1.0)
        self.assertEqual(weights["paired_answer"], -1.0)
        self.assertEqual(weights["independent_evidence"], -1.0)
        self.assertEqual(weights["independent_answer"], 1.0)

    def test_non_applicable_id_metric_and_saturation_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "analysis"
            arguments = []
            for run_name in analysis.RUN_ORDER:
                path = root / f"{run_name}.jsonl"
                rows = rows_for_run(run_name, answer_value=run_name != "base", evidence_ids_applicable=False)
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                arguments.extend(["--run", f"{run_name}={path}"])
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    *arguments,
                    "--output-dir",
                    str(output),
                    "--bootstrap-replicates",
                    "100",
                    "--seed",
                    "9",
                    "--expected-clusters",
                    "2",
                ]
                self.assertEqual(analysis.main(), 0)
            finally:
                sys.argv = old_argv
            report = json.loads((output / "factorial_analysis.json").read_text())
            self.assertTrue(
                report["benchmark_discrimination"]["saturated_at_98pct_and_lt_2pp_range"]
            )
            self.assertIsNone(
                report["run_summary_intervals"]["base"]["evidence_ids_correct"]["estimate"]
            )
            self.assertEqual(report["bootstrap"]["replicates"], 100)
            self.assertEqual(report["bootstrap"]["cluster_key"], "metadata.case_id")
            self.assertEqual(report["bootstrap"]["cluster_strata_key"], "task")
            self.assertEqual(report["bootstrap"]["cluster_count"], 2)
            self.assertEqual(report["bootstrap"]["expected_clusters"], 2)
            self.assertTrue(report["bootstrap"]["auto_nolima_cluster_mode"])
            self.assertTrue((output / "factorial_contrasts.csv").is_file())
            self.assertTrue((output / "paired_bootstrap_indices.jsonl.gz").is_file())

    def test_non_applicable_quotes_and_invalid_json_do_not_crash_or_fake_consistency(
        self,
    ) -> None:
        rows = rows_for_run(
            "base",
            answer_value=False,
            evidence_quotes_applicable=False,
        )
        for row in rows:
            row["parsed"] = None
        grouped, conditions = analysis.build_group_tables(
            {run: [dict(row) for row in rows] for run in analysis.RUN_ORDER}
        )
        summary, _ = analysis.summarize_selection(grouped["base"], conditions)
        self.assertIsNone(summary["evidence_quotes_correct"])
        self.assertIsNone(summary["mean_worst_quote_accuracy"])
        self.assertIsNone(summary["mean_quote_position_gap"])
        self.assertEqual(summary["same_answer_across_positions"], 0.0)

    def test_cluster_metadata_is_joined_by_exact_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl"
            source_rows = rows_for_run("source", answer_value=True)
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in source_rows),
                encoding="utf-8",
            )
            result_rows = [
                {key: value for key, value in row.items() if key != "metadata"}
                for row in source_rows
            ]
            rows_by_run = {
                run: [dict(row) for row in result_rows]
                for run in analysis.RUN_ORDER
            }
            digest = analysis.attach_cluster_metadata(rows_by_run, source)
            self.assertEqual(len(digest), 64)
            self.assertEqual(
                rows_by_run["base"][0]["metadata"]["case_id"],
                "case-0",
            )
            rows_by_run["base"].pop()
            with self.assertRaises(SystemExit):
                analysis.attach_cluster_metadata(rows_by_run, source)


if __name__ == "__main__":
    unittest.main()
