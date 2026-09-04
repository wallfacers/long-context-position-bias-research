from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_factorial_results.py"
SPEC = importlib.util.spec_from_file_location("plot_factorial_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plots = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plots
SPEC.loader.exec_module(plots)


class FactorialPlotTest(unittest.TestCase):
    def test_writes_vector_raster_and_accessibility_metadata(self) -> None:
        profiles = []
        for run_index, run_name in enumerate(plots.SUMMARY_RUN_ORDER):
            for position_index, position in enumerate(plots.POSITIONS):
                value = min(0.99, 0.35 + run_index * 0.08 + position_index * 0.01)
                profiles.append(
                    {
                        "run_name": run_name,
                        "task": "nolima_hard",
                        "target_tokens": 8192,
                        "position_label": position,
                        "answer_accuracy": value,
                        "ci95_low": max(0.0, value - 0.03),
                        "ci95_high": min(1.0, value + 0.03),
                    }
                )
        intervals = {
            run_name: {
                statistic: {
                    "estimate": 0.7,
                    "ci95_low": 0.65,
                    "ci95_high": 0.75,
                }
                for statistic in (
                    "answer_correct",
                    "mean_worst_answer_accuracy",
                    "mean_answer_position_gap",
                    "evidence_quotes_correct",
                )
            }
            for run_name in plots.SUMMARY_RUN_ORDER
        }
        report = {
            "schema_version": "matched-factorial-analysis-v1",
            "position_profiles": profiles,
            "run_summary_intervals": intervals,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_path = root / "analysis.json"
            output_dir = root / "figures"
            analysis_path.write_text(json.dumps(report), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--analysis",
                    str(analysis_path),
                    "--output-dir",
                    str(output_dir),
                ]
                self.assertEqual(plots.main(), 0)
            finally:
                sys.argv = old_argv
            for name in (
                "factorial_position_curves.svg",
                "factorial_position_curves.pdf",
                "factorial_position_curves.png",
                "factorial_summary.svg",
                "factorial_summary.pdf",
                "factorial_summary.png",
                "figures.metadata.json",
            ):
                self.assertGreater((output_dir / name).stat().st_size, 0)
            metadata = json.loads((output_dir / "figures.metadata.json").read_text())
            self.assertIn("position_curves_alt", metadata["accessibility"])


if __name__ == "__main__":
    unittest.main()
