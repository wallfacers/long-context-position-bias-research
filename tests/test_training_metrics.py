from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_training_metrics", ROOT / "scripts" / "export_training_metrics.py"
)
assert SPEC and SPEC.loader
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class TrainingMetricExportTests(unittest.TestCase):
    def test_metric_rows_deduplicate_and_validate(self) -> None:
        state = {
            "log_history": [
                {"step": 1, "loss": 0.4, "learning_rate": 0.0},
                {"step": 2, "loss": 0.3, "learning_rate": 1e-5},
                {"step": 2, "loss": 0.2, "learning_rate": 2e-5},
                {"train_runtime": 12.0, "epoch": 1.0, "step": 2},
            ]
        }
        rows, duplicates = METRICS.metric_rows(state, "paired_answer")
        self.assertEqual([row["step"] for row in rows], [1, 2])
        self.assertEqual(rows[-1]["loss"], 0.2)
        self.assertEqual(duplicates, 1)

    def test_latest_checkpoint_uses_numeric_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (9, 100, 20):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
            step, path = METRICS.latest_checkpoint(root)
            self.assertEqual(step, 100)
            self.assertEqual(path.name, "checkpoint-100")

    def test_summary_flags_saturated_training_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "TRAINING_COMPLETE.json").write_text(
                json.dumps({"global_step": 3}), encoding="utf-8"
            )
            rows = [
                {
                    "variant": "paired_evidence",
                    "step": step,
                    "epoch": step / 3,
                    "loss": 1e-6,
                    "grad_norm": 1e-5,
                    "learning_rate": 1e-4,
                    "entropy": 1e-5,
                    "num_tokens": step * 100,
                    "mean_token_accuracy": 1.0,
                    "eval_loss": None,
                }
                for step in (1, 2, 3)
            ]
            summary = METRICS.summarize(
                variant="paired_evidence",
                rows=rows,
                duplicate_steps=0,
                state={"global_step": 3},
                checkpoint_step=3,
                expected_steps=3,
                output_dir=output,
            )
            self.assertTrue(summary["training_complete"])
            self.assertEqual(summary["missing_steps"], [])
            self.assertEqual(len(summary["warnings"]), 2)

    def test_summary_accepts_explicit_canary_completion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "CANARY_COMPLETE.json").write_text(
                json.dumps({"global_step": 1}), encoding="utf-8"
            )
            summary = METRICS.summarize(
                variant="paired_answer",
                rows=[
                    {
                        "variant": "paired_answer",
                        "step": 1,
                        "epoch": 1.0,
                        "loss": 0.1,
                        "grad_norm": 0.2,
                        "learning_rate": 1e-4,
                        "entropy": 0.3,
                        "num_tokens": 100,
                        "mean_token_accuracy": 0.9,
                        "eval_loss": None,
                    }
                ],
                duplicate_steps=0,
                state={"global_step": 1},
                checkpoint_step=1,
                expected_steps=1,
                output_dir=output,
                completion_record_name="CANARY_COMPLETE.json",
            )
            self.assertTrue(summary["training_complete"])
            self.assertEqual(
                summary["completion_record_name"], "CANARY_COMPLETE.json"
            )


if __name__ == "__main__":
    unittest.main()
