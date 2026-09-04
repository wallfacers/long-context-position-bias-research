from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_pilot_report.py"
SPEC = importlib.util.spec_from_file_location("render_pilot_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def statistic(value: float) -> dict[str, float]:
    return {
        "estimate": value,
        "ci95_low": value - 0.01,
        "ci95_high": value + 0.01,
        "bootstrap_p_two_sided": 0.1,
        "holm_adjusted_p": 0.2,
    }


class PilotReportTests(unittest.TestCase):
    def test_report_requires_complete_validated_matrix_and_renders_guardrails(self) -> None:
        runs = reporter.RUN_ORDER
        summary = {
            name: {
                "answer_correct": 0.8,
                "evidence_ids_correct": 0.7,
                "evidence_quotes_correct": 0.7,
                "all_predicted_quotes_supported": 0.9,
                "valid_json": 1.0,
                "mean_worst_position_accuracy": 0.7,
                "mean_position_gap": 0.1,
                "mean_edge_accuracy": 0.8,
                "mean_middle_accuracy": 0.75,
                "mean_middle_penalty": 0.05,
            }
            for name in runs
        }
        stats = tuple(summary["base"])
        contrast_names = (
            "paired_minus_independent_main_effect",
            "evidence_minus_answer_main_effect",
            "pairing_x_supervision_interaction",
        )
        contrasts = {
            name: {"statistics": {key: statistic(0.01) for key in stats}}
            for name in contrast_names
        }
        checks = {
            "gap_reduction_at_least_50pct": True,
            "worst_position_gain_at_least_10pp": True,
            "mean_answer_drop_no_more_than_2pp": True,
            "edge_accuracy_drop_no_more_than_2pp": True,
            "valid_json_at_least_99pct": True,
        }
        analysis = {
            "schema_version": "position-ablation-analysis-v1",
            "bootstrap": {"replicates": 2000, "seed": 20260825},
            "rows_per_run": {name: 4200 for name in runs},
            "run_summaries": summary,
            "generation_diagnostics": {
                name: {"finish_reason_length_rate": 0.01} for name in runs
            },
            "run_summary_intervals": {
                name: {key: statistic(value) for key, value in values.items()}
                for name, values in summary.items()
            },
            "contrasts": contrasts,
            "exploratory_screening": {
                "results": [
                    {
                        "run_name": name,
                        "gap_reduction_fraction": 0.5,
                        "worst_position_gain": 0.1,
                        "mean_answer_delta": 0.0,
                        "edge_accuracy_delta": 0.0,
                        "checks": checks,
                        "passes_all_exploratory_checks": True,
                    }
                    for name in runs[1:]
                ]
            },
        }
        validation = {
            "status": "validated",
            "matrix": {"total_samples": 21000},
            "execution": {
                "wall_hours": 10.0,
                "hourly_rate_cny": 2.78,
                "estimated_cost_cny": 27.8,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_path = root / "analysis.json"
            validation_path = root / "validation.json"
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            rendered = reporter.render(analysis_path, validation_path)
        self.assertIn("21,000 条预测", rendered)
        self.assertIn("首尾≥−2pp", rendered)
        self.assertIn("Evidence − answer 主效应", rendered)
        self.assertIn("单 seed", rendered)


if __name__ == "__main__":
    unittest.main()
