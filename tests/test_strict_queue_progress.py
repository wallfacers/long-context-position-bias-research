from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_strict_queue_progress.py"
SPEC = importlib.util.spec_from_file_location("summarize_strict_queue_progress", SCRIPT)
progress = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(progress)


class StrictQueueProgressTest(unittest.TestCase):
    def test_queue_health_is_fail_closed_and_distinguishes_stalls(self):
        self.assertEqual(
            progress.scheduler_daemon_lines(
                "1 0 00:10 Ss /usr/sbin/cron -f\n2 1 00:10 S python worker.py"
            ),
            ["1 0 00:10 Ss /usr/sbin/cron -f"],
        )
        self.assertEqual(
            progress.classify_queue_health(
                {"top": "running stage=qwen", "qwen": "failed exit_code=1"},
                True,
            ),
            "failed",
        )
        self.assertEqual(
            progress.classify_queue_health({"top": "running stage=qwen"}, False),
            "stalled",
        )
        self.assertEqual(
            progress.classify_queue_health(
                {
                    "top": "running stage=qwen",
                    "qwen": "running stage=block96_training",
                    "mistral": "failed exit_code=1",
                },
                True,
            ),
            "running",
        )
        self.assertEqual(
            progress.classify_queue_health({"top": "validated exit_code=0"}, False),
            "validated",
        )

    def test_registry_matches_block96_labeled_runners(self):
        self.assertIn(
            "independent_evidence_id_block96",
            progress.EVAL_WORKLOADS["qwen_block96_mmlu"][1],
        )
        self.assertIn(
            "paired_evidence_block96",
            progress.EVAL_WORKLOADS["mistral_block96_nolima_mechanisms"][1],
        )
        self.assertIn(
            "independent_evidence_id",
            progress.EVAL_WORKLOADS["qwen_block96_ifeval"][1],
        )

    def make_completed_condition(self, root: Path) -> None:
        output = root / "outputs/qwen_block96/seed_20260825/independent_answer"
        checkpoint = output / "checkpoint-96"
        checkpoint.mkdir(parents=True)
        rows = [
            {
                "step": step,
                "loss": 1e-5,
                "learning_rate": 2e-4,
                "grad_norm": 1e-3,
                "mean_token_accuracy": 1.0,
            }
            for step in range(1, 97)
        ]
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 96, "log_history": rows}), encoding="utf-8"
        )
        (output / "CANARY_COMPLETE.json").write_text(
            json.dumps(
                {
                    "schema_version": "qlora-result-v1",
                    "global_step": 96,
                    "elapsed_seconds_this_invocation": 240,
                    "seconds_per_step_this_invocation": 2.5,
                }
            ),
            encoding="utf-8",
        )

    def test_completed_condition_requires_exact_full_metric_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_completed_condition(root)
            record = progress.completed_training_record(
                root, "qwen_block96", 20260825, "independent_answer"
            )
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["metric_rows"], 96)
            self.assertTrue(record["all_metrics_finite"])
            self.assertTrue(record["training_distribution_saturated"])

            state = root / "outputs/qwen_block96/seed_20260825/independent_answer/checkpoint-96/trainer_state.json"
            payload = json.loads(state.read_text())
            payload["log_history"].pop()
            state.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(
                progress.completed_training_record(
                    root, "qwen_block96", 20260825, "independent_answer"
                )
            )

    def test_one_shot_cli_emits_deterministic_snapshot_without_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_completed_condition(root)
            (root / "configs").mkdir()
            budget = json.loads(
                (ROOT / "configs/autodl_strict_block96_budget.json").read_text()
            )
            (root / "configs/autodl_strict_block96_budget.json").write_text(
                json.dumps(budget), encoding="utf-8"
            )
            output = root.parent / f"{root.name}-snapshot.json"
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--project-root",
                    str(root),
                    "--status-root",
                    str(root / "status"),
                    "--now",
                    "2026-08-29T16:18:07+08:00",
                    "--no-system-probes",
                    "--output",
                    str(output),
                ]
                self.assertEqual(progress.main(), 0)
            finally:
                sys.argv = old_argv
            payload = json.loads(output.read_text())
            self.assertEqual(payload["schema_version"], "strict-queue-progress-snapshot-v1")
            self.assertTrue(payload["read_only_one_shot"])
            self.assertEqual(payload["queue_health"], "unknown")
            self.assertEqual(payload["training"]["conditions_complete"], 1)
            self.assertIsNone(payload["gpu"])
            self.assertIsNone(payload["runtime_safety"])
            self.assertGreater(payload["estimate"]["remaining_gpu_hours"], 0)
            self.assertLess(
                payload["estimate"]["remaining_gpu_hours_measured_calibrated"],
                payload["estimate"]["remaining_gpu_hours"],
            )
            self.assertEqual(
                payload["estimate"]["measured_training_seconds_per_step_by_family"][
                    "qwen_block96"
                ],
                2.5,
            )


if __name__ == "__main__":
    unittest.main()
