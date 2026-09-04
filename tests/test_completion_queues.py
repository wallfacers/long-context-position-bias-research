from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
QUEUE_SCRIPTS = (
    "run_autodl_qwen_seed1_completion_queue.sh",
    "run_autodl_qwen_confirmatory_queue.sh",
    "run_autodl_mistral_completion_queue.sh",
    "run_autodl_full_completion_queue.sh",
    "run_autodl_qwen_block96_completion_queue.sh",
    "run_autodl_mistral_block96_completion_queue.sh",
    "run_autodl_strict_block96_full_queue.sh",
)


class CompletionQueueTest(unittest.TestCase):
    def test_actual_work_queues_have_no_timer_watcher_or_power_action(self):
        forbidden = re.compile(
            r"\b(?:shutdown|poweroff|halt|reboot|systemctl|sleep|crontab|at|watch)\b"
        )
        for name in QUEUE_SCRIPTS:
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            executable = "\n".join(
                line for line in text.splitlines() if not line.lstrip().startswith("echo ")
            )
            self.assertIsNone(forbidden.search(executable), name)

    def test_seed1_queue_contains_every_frozen_suite(self):
        text = (ROOT / "scripts" / QUEUE_SCRIPTS[0]).read_text(encoding="utf-8")
        for runner in (
            "run_autodl_nolima_gate.sh",
            "run_autodl_longbench_transfer.sh",
            "run_autodl_mmlu_regression.sh",
            "run_autodl_ifeval_regression.sh",
            "run_autodl_nolima_mechanisms.sh",
        ):
            self.assertIn(runner, text)

    def test_confirmatory_queues_freeze_seed_sets(self):
        qwen = (ROOT / "scripts" / QUEUE_SCRIPTS[1]).read_text(encoding="utf-8")
        mistral = (ROOT / "scripts" / QUEUE_SCRIPTS[2]).read_text(encoding="utf-8")
        self.assertIn('seeds_csv="20260826,20260827"', qwen)
        self.assertIn('seeds_csv="20260825,20260826,20260827"', mistral)
        self.assertIn("generate_paper_results.py", mistral)
        for argument in (
            "--qwen-mmlu",
            "--qwen-ifeval",
            "--qwen-mechanisms",
            "--mistral-mmlu",
            "--mistral-ifeval",
            "--mistral-mechanisms",
            "--qwen-exploratory-rule",
        ):
            self.assertIn(argument, mistral)
        self.assertIn('"artifact_sha256"', mistral)

    def test_training_queue_activates_dedicated_environment(self):
        text = (ROOT / "scripts" / "run_autodl_fixed100_training_queue.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("POSITION_BIAS_TRAIN_VENV", text)
        self.assertIn('source "$train_venv/bin/activate"', text)

    def test_block96_queue_requires_realized_subset_audits(self):
        text = (
            ROOT / "scripts" / "run_autodl_qwen_block96_completion_queue.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("audit_realized_training_subset.py", text)
        self.assertIn("--fixed-steps 96", text)
        self.assertIn("--checkpoint-name \"$checkpoint_name\"", text)
        self.assertIn("qwen_fixed100_realized_subset.json", text)
        self.assertIn("qwen_block96_realized_subset.json", text)
        self.assertIn("Historical fixed-100 correction premise changed unexpectedly", text)
        self.assertIn("retrain_from_materialized_block_complete_subsets", text)
        self.assertIn("Strict materialized Qwen completion lineage hash mismatch", text)
        self.assertIn("Strict materialized Qwen seed record failed", text)
        for field in ("manifest_sha256", "matched_audit_sha256", "selection_sha256"):
            self.assertIn(field, text)
        mistral = (
            ROOT / "scripts" / "run_autodl_mistral_block96_completion_queue.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("materialize_block_complete_sft.py", mistral)
        self.assertIn("mistral_block96_realized_subset.json", mistral)
        self.assertIn("prospective_under_corrected_protocol", mistral)

    def test_strict_full_queue_orders_qwen_before_mistral(self):
        text = (
            ROOT / "scripts" / "run_autodl_strict_block96_full_queue.sh"
        ).read_text(encoding="utf-8")
        qwen = text.index("run_autodl_qwen_block96_completion_queue.sh")
        mistral = text.index("run_autodl_mistral_block96_completion_queue.sh")
        self.assertLess(qwen, mistral)
        self.assertIn("historical_fixed100_primary_eligible", text)
        self.assertIn("strict_block96_failure_case_catalog_audit.json", text)
        self.assertIn("--expected-training-step 96", text)
        self.assertIn("--require-evidence-label compute_accounting", text)
        self.assertIn("full_paper_evidence_manifest.json", text)
        self.assertIn('hashes = payload.get("artifact_sha256", {})', text)

    def test_fixed100_scheduler_horizon_and_callback_are_frozen(self):
        variant_runner = (ROOT / "scripts" / "run_sft_variant.sh").read_text(
            encoding="utf-8"
        )
        trainer = (ROOT / "scripts" / "train_qlora.py").read_text(encoding="utf-8")
        diagnostic_runner = (
            ROOT / "scripts" / "run_autodl_fixed100_training_queue.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--max-steps 2000", variant_runner)
        self.assertIn('--stop-after-steps "$FIXED_STEPS"', variant_runner)
        self.assertIn("FIXED_STEPS=100", variant_runner)
        self.assertIn('parser.add_argument("--warmup-ratio", type=float, default=0.03)', trainer)
        self.assertIn("--scheduler-horizon 2000", diagnostic_runner)
        self.assertIn("--warmup-steps 60", diagnostic_runner)

    def test_full_queue_preserves_required_stage_order(self):
        text = (ROOT / "scripts" / QUEUE_SCRIPTS[3]).read_text(encoding="utf-8")
        stages = (
            "finalize_qwen_seed1_formal.sh",
            "run_autodl_qwen_seed1_completion_queue.sh",
            "run_autodl_qwen_confirmatory_queue.sh",
            "run_autodl_mistral_completion_queue.sh",
        )
        offsets = [text.index(stage) for stage in stages]
        self.assertEqual(offsets, sorted(offsets))


if __name__ == "__main__":
    unittest.main()
