from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_position_ablation.py"
SPEC = importlib.util.spec_from_file_location("plot_position_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plots = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plots
SPEC.loader.exec_module(plots)


class AblationPlotTests(unittest.TestCase):
    def test_render_writes_vector_raster_and_accessibility_metadata(self) -> None:
        profiles = []
        for run_index, run_name in enumerate(plots.RUN_ORDER):
            for task in ("kv", "two_hop"):
                for length in (8192, 32768):
                    for position in plots.POSITIONS:
                        value = 0.5 + run_index * 0.1
                        profiles.append(
                            {
                                "run_name": run_name,
                                "task": task,
                                "target_tokens": length,
                                "position_label": position,
                                "answer_accuracy": value,
                                "ci95_low": value - 0.05,
                                "ci95_high": value + 0.05,
                            }
                        )
        intervals = {
            run_name: {
                statistic: {
                    "estimate": 0.5,
                    "ci95_low": 0.45,
                    "ci95_high": 0.55,
                }
                for statistic in (
                    "answer_correct",
                    "mean_worst_position_accuracy",
                    "mean_position_gap",
                )
            }
            for run_name in plots.RUN_ORDER
        }
        report = {
            "schema_version": "position-ablation-analysis-v1",
            "position_profiles": profiles,
            "run_summary_intervals": intervals,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_path = root / "analysis.json"
            figures = root / "figures"
            analysis_path.write_text(json.dumps(report), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--analysis",
                    str(analysis_path),
                    "--output-dir",
                    str(figures),
                ]
                self.assertEqual(plots.main(), 0)
            finally:
                sys.argv = old_argv
            for name in (
                "position_curves.svg",
                "position_curves.pdf",
                "position_curves.png",
                "ablation_summary.svg",
                "ablation_summary.pdf",
                "ablation_summary.png",
                "figures.metadata.json",
            ):
                self.assertGreater((figures / name).stat().st_size, 0)
            metadata = json.loads(
                (figures / "figures.metadata.json").read_text(encoding="utf-8")
            )
            self.assertIn("position_curves_alt", metadata["accessibility"])

            # Regression guard for the paper figure header: the overall title,
            # one-row legend, and first panel title must occupy distinct bands.
            root_element = ET.parse(figures / "position_curves.svg").getroot()
            text_y: dict[str, float] = {}
            for element in root_element.iter():
                if not element.tag.endswith("text"):
                    continue
                label = "".join(element.itertext()).strip()
                if not label:
                    continue
                if "y" in element.attrib:
                    text_y.setdefault(label, float(element.attrib["y"]))
                    continue
                match = re.search(
                    r"translate\([^ ]+ ([^\)]+)\)", element.attrib.get("transform", "")
                )
                if match:
                    text_y.setdefault(label, float(match.group(1)))
            title_y = text_y["Position robustness across the 2×2 training ablation"]
            legend_y = text_y["Base"]
            panel_y = text_y["Key-value retrieval · 8K"]
            self.assertGreater(legend_y - title_y, 10)
            self.assertGreater(panel_y - legend_y, 10)


if __name__ == "__main__":
    unittest.main()
