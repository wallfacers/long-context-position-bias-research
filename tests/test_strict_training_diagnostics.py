from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plot_strict_training_diagnostics.py"
FAMILIES = ("Qwen2.5-7B", "Mistral-7B-v0.3")
SEEDS = (20260825, 20260826, 20260827)
VARIANTS = (
    "independent_answer",
    "paired_answer",
    "independent_evidence_id",
    "paired_evidence_id",
    "independent_evidence",
    "paired_evidence",
)


class StrictTrainingDiagnosticsTest(unittest.TestCase):
    def write_metrics(self, path: Path, family_index: int, seed_index: int) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
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
            for variant_index, variant in enumerate(VARIANTS):
                for step in range(1, 97):
                    writer.writerow(
                        {
                            "variant": variant,
                            "step": step,
                            "loss": (1 + family_index + seed_index / 10 + variant_index / 20) / step,
                            "learning_rate": min(step, 60) / 60 * 2e-4,
                            "grad_norm": (2 + seed_index / 10 + variant_index / 20) / step,
                            "mean_token_accuracy": min(1.0, 0.5 + step / 100),
                        }
                    )

    def test_cli_renders_auditable_cross_seed_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = []
            for family_index, family in enumerate(FAMILIES):
                for seed_index, seed in enumerate(SEEDS):
                    path = root / f"{family_index}-{seed}.csv"
                    self.write_metrics(path, family_index, seed_index)
                    arguments.extend(("--metrics", f"{family}:{seed}:{path}"))
            output = root / "figures"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    *arguments,
                    "--output-dir",
                    str(output),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            manifest = json.loads(
                (output / "strict_training_diagnostics.manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "validated")
            self.assertEqual(manifest["families"], list(FAMILIES))
            self.assertEqual(manifest["seeds"], list(SEEDS))
            self.assertEqual(manifest["table_rows"], 2 * 6 * 96 * 4)
            self.assertEqual(len(manifest["sources"]), 6)
            for record in manifest["outputs"].values():
                path = Path(record["path"])
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            alt = (output / "strict_training_diagnostics.alt.txt").read_text()
            self.assertIn("translucent envelope is the seed minimum-to-maximum range", alt)
            self.assertIn("Exact plotted values", alt)


if __name__ == "__main__":
    unittest.main()
