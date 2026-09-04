import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_compute_accounting.py"
SPEC = importlib.util.spec_from_file_location("summarize_compute_accounting", SCRIPT)
accounting = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(accounting)


class ComputeAccountingTest(unittest.TestCase):
    def test_sums_active_time_and_deduplicates_reused_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "outputs" / "cell"
            eval_a = root / "results" / "a"
            eval_b = root / "results" / "b"
            train.mkdir(parents=True)
            eval_a.mkdir(parents=True)
            eval_b.mkdir(parents=True)
            (train / "CANARY_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "schema_version": "qlora-result-v1",
                        "run_id": "train-1",
                        "finished_at": "2026-08-29T00:01:00Z",
                        "global_step": 100,
                        "elapsed_seconds_this_invocation": 100.0,
                    }
                ),
                encoding="utf-8",
            )
            run = {
                "schema_version": "vllm-eval-run-v1",
                "status": "selection_complete",
                "started_at": "2026-08-29T00:02:00Z",
                "last_finished_at": "2026-08-29T00:04:00Z",
                "model": "models/model-a",
                "adapter_sha256": None,
                "data_sha256": "data-hash",
                "selection_sha256": "selection-hash",
                "selected_samples": 20,
                "elapsed_seconds_total": 200.0,
            }
            for parent in (eval_a, eval_b):
                (parent / "base.jsonl.run.json").write_text(
                    json.dumps(run), encoding="utf-8"
                )
            output = root / "results" / "compute.json"
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--project-root",
                    str(root),
                    "--training-root",
                    str(train.parent),
                    "--eval-root",
                    str(eval_a),
                    "--eval-root",
                    str(eval_b),
                    "--hourly-rate",
                    "3.0",
                    "--output",
                    str(output),
                ]
                self.assertEqual(accounting.main(), 0)
            finally:
                sys.argv = old_argv
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["unique_events"], 2)
            self.assertEqual(report["deduplicated_event_copies"], 1)
            self.assertEqual(report["total_active_gpu_seconds"], 300.0)
            self.assertAlmostEqual(report["active_gpu_cost_lower_bound_cny"], 0.25)
            self.assertEqual(report["by_kind"]["training"]["units"], 100)
            self.assertEqual(report["by_kind"]["evaluation"]["units"], 20)
            self.assertEqual(report["input_roots"]["training"], ["outputs"])
            self.assertTrue(all(item["source_sha256"] for item in report["events"]))

    def test_accepts_an_explicit_block96_training_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "outputs" / "cell"
            train.mkdir(parents=True)
            (train / "CANARY_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "schema_version": "qlora-result-v1",
                        "run_id": "strict-96",
                        "finished_at": "2026-08-29T00:01:00Z",
                        "global_step": 96,
                        "elapsed_seconds_this_invocation": 96.0,
                    }
                ),
                encoding="utf-8",
            )
            records, duplicates = accounting.collect(root, [train], [], 96)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["units"], 96)
            self.assertEqual(duplicates, [])

    def test_accepts_completed_ifeval_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "results" / "ifeval" / "generations"
            evaluation.mkdir(parents=True)
            (evaluation / "base.jsonl.run.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ifeval-vllm-run-v1",
                        "status": "selection_complete",
                        "started_at": "2026-08-29T00:02:00Z",
                        "last_finished_at": "2026-08-29T00:04:00Z",
                        "model": "models/model-a",
                        "adapter_sha256": None,
                        "data_sha256": "data-hash",
                        "selection_sha256": "selection-hash",
                        "samples": 541,
                        "selected_samples": 541,
                        "elapsed_seconds_total": 200.0,
                    }
                ),
                encoding="utf-8",
            )
            records, duplicates = accounting.collect(root, [], [evaluation.parent])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["units"], 541)
            self.assertEqual(duplicates, [])


if __name__ == "__main__":
    unittest.main()
