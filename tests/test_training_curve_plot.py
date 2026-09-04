from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_training_curves.py"
VARIANTS = (
    "independent_answer",
    "independent_evidence_id",
    "independent_evidence",
    "paired_answer",
    "paired_evidence_id",
    "paired_evidence",
)


class TrainingCurvePlotTests(unittest.TestCase):
    def test_plot_records_scheduler_semantics_and_accessible_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics = Path(temporary)
            metrics = diagnostics / "training_metrics.csv"
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "variant",
                        "step",
                        "loss",
                        "learning_rate",
                        "grad_norm",
                        "mean_token_accuracy",
                    ),
                )
                writer.writeheader()
                for variant in VARIANTS:
                    for step in (1, 2, 3):
                        writer.writerow(
                            {
                                "variant": variant,
                                "step": step,
                                "loss": 1.0 / (10 * step),
                                "learning_rate": step * 1e-5,
                                "grad_norm": 1.0 / step,
                                "mean_token_accuracy": 0.9 + step / 100,
                            }
                        )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--diagnostics-dir",
                    str(diagnostics),
                    "--scheduler-horizon",
                    "2000",
                    "--warmup-steps",
                    "60",
                    "--dpi",
                    "72",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            figures = diagnostics / "figures"
            metadata = json.loads(
                (figures / "training_curves.figure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema_version"], "training-curve-figure-v2")
            self.assertEqual(metadata["scheduler_horizon_steps"], 2000)
            self.assertEqual(metadata["warmup_steps"], 60)
            self.assertEqual(
                metadata["visual_encoding"]["line_style"],
                "position construction (independent dashed, paired solid)",
            )
            self.assertEqual(metadata["rows_by_variant"], {name: 3 for name in VARIANTS})
            alt_text = (figures / "training_curves.alt.txt").read_text(encoding="utf-8")
            self.assertIn("2,000-step cosine horizon", alt_text)
            self.assertIn("Dashed lines denote independent", alt_text)
            for suffix in ("pdf", "png", "svg"):
                self.assertTrue((figures / f"training_curves.{suffix}").is_file())


if __name__ == "__main__":
    unittest.main()
