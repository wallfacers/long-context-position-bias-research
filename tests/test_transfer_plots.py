from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_transfer_results.py"
SPEC = importlib.util.spec_from_file_location("plot_transfer_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plots = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plots
SPEC.loader.exec_module(plots)


class TransferPlotTest(unittest.TestCase):
    def test_writes_accessible_vector_and_raster_figures(self) -> None:
        intervals = {
            slice_name: {
                run_name: {
                    "estimate": 0.4 + index * 0.05,
                    "ci95_low": 0.37 + index * 0.05,
                    "ci95_high": 0.43 + index * 0.05,
                    "n": 20,
                }
                for index, run_name in enumerate(plots.RUN_ORDER)
            }
            for slice_name in plots.SLICE_ORDER
        }
        report = {
            "schema_version": "natural-transfer-analysis-v1",
            "run_intervals": intervals,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "analysis.json"
            output = root / "figures"
            source.write_text(json.dumps(report), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [str(SCRIPT), "--analysis", str(source), "--output-dir", str(output)]
                self.assertEqual(plots.main(), 0)
            finally:
                sys.argv = old_argv
            for name in (
                "natural_transfer_heatmap.png",
                "natural_transfer_heatmap.svg",
                "natural_transfer_heatmap.pdf",
                "natural_transfer_intervals.png",
                "natural_transfer_intervals.svg",
                "natural_transfer_intervals.pdf",
                "figures.metadata.json",
            ):
                self.assertGreater((output / name).stat().st_size, 0)
            metadata = json.loads((output / "figures.metadata.json").read_text())
            self.assertIn("heatmap_alt", metadata["accessibility"])


if __name__ == "__main__":
    unittest.main()
