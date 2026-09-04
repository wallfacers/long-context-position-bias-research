import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NoLiMaMechanismPlotTest(unittest.TestCase):
    def test_writes_three_formats_and_accessibility_metadata(self):
        runs = (
            "base",
            "independent_answer",
            "independent_evidence",
            "paired_answer",
            "paired_evidence",
        )
        metrics = (
            "free_answer",
            "oracle_long_answer",
            "oracle_short_answer",
            "free_quote",
            "locate_quote",
        )
        report = {
            "schema_version": "nolima-mechanism-analysis-v1",
            "run_intervals": {
                run: {
                    metric: {"estimate": 0.5, "ci95_low": 0.4, "ci95_high": 0.6}
                    for metric in metrics
                }
                for run in runs
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps(report), encoding="utf-8")
            output = root / "figures"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/plot_nolima_mechanisms.py"),
                    "--analysis",
                    str(analysis),
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
            for extension in ("png", "svg", "pdf"):
                self.assertTrue(
                    (output / f"nolima_mechanism_decomposition.{extension}").is_file()
                )
            metadata = json.loads(
                (output / "figures.metadata.json").read_text(encoding="utf-8")
            )
            self.assertIn("alt", metadata["accessibility"])


if __name__ == "__main__":
    unittest.main()
