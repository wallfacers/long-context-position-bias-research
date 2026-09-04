from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_bias_research.synthetic_data import (
    generate_position_equivalent_group,
    iter_matched_training_bank,
    iter_synthetic_samples,
)
from position_bias_research.tokenization import WhitespaceTokenCounter


PREPARE_SCRIPT = ROOT / "scripts" / "prepare_sft_variants.py"
SPEC = importlib.util.spec_from_file_location("prepare_sft_variants", PREPARE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def load_script(module_name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare_matched = load_script(
    "prepare_matched_sft_variants", "prepare_matched_sft_variants.py"
)
audit_matched = load_script(
    "audit_matched_training_design", "audit_matched_training_design.py"
)
prepare_diagnostics = load_script(
    "prepare_diagnostic_eval", "prepare_diagnostic_eval.py"
)


class SyntheticDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = WhitespaceTokenCounter()

    def test_position_group_preserves_fact_and_filler(self) -> None:
        group = generate_position_equivalent_group(
            split="train",
            group_index=0,
            task="two_hop",
            filler_type="answer_bearing",
            target_tokens=512,
            positions=(0.0, 0.25, 0.5, 1.0),
            seed=7,
            token_counter=self.counter,
            words_per_document=16,
        )
        self.assertEqual(len(group), 4)
        self.assertEqual(len({sample["group_id"] for sample in group}), 1)
        self.assertEqual(len({str(sample["target"]) for sample in group}), 1)
        self.assertEqual(
            len({sample["metadata"]["filler_fingerprint"] for sample in group}), 1
        )
        self.assertEqual(
            [sample["target_position"] for sample in group], [0.0, 0.25, 0.5, 1.0]
        )
        for sample in group:
            self.assertLess(abs(sample["actual_tokens"] - 512) / 512, 0.12)
            for quote in sample["target"]["evidence_quotes"]:
                self.assertIn(quote, sample["prompt"])

    def test_four_sft_variants_have_equal_sequence_and_token_budgets(self) -> None:
        rows = list(
            iter_synthetic_samples(
                split="train",
                groups_per_condition=5,
                tasks=("kv", "two_hop"),
                filler_types=("neutral",),
                target_lengths=(512,),
                positions=(0.0, 0.25, 0.5, 1.0),
                seed=11,
                token_counter=self.counter,
                words_per_document=16,
            )
        )
        variants = prepare.build_variants(
            rows,
            paired_groups=2,
            independent_groups=8,
            seed=11,
            max_token_budget_gap=0.12,
        )
        self.assertEqual(set(variants), set(prepare.VARIANTS))
        self.assertEqual({len(samples) for samples in variants.values()}, {8})
        token_totals = {
            sum(sample["metadata"]["actual_tokens"] for sample in samples)
            for samples in variants.values()
        }
        self.assertEqual(len(token_totals), 1)
        for name, samples in variants.items():
            expected_evidence = name.endswith("evidence")
            for sample in samples:
                self.assertEqual(
                    sample["metadata"]["include_evidence_supervision"],
                    expected_evidence,
                )
                response = json.loads(sample["messages"][-1]["content"])
                self.assertEqual(
                    bool(response["evidence_ids"] or response["evidence_quotes"]),
                    expected_evidence,
                )
                self.assertEqual(len(sample["messages"]), 3)

    def test_matched_bank_separates_fact_and_filler_replica_seeds(self) -> None:
        rows = list(
            iter_matched_training_bank(
                split="train",
                facts_per_condition=2,
                replicas_per_fact=4,
                tasks=("kv",),
                filler_types=("neutral",),
                target_lengths=(512,),
                positions=(0.0, 0.25, 0.5, 1.0),
                seed=17,
                token_counter=self.counter,
                words_per_document=16,
            )
        )
        self.assertEqual(len(rows), 32)
        facts: dict[str, list[dict]] = {}
        for row in rows:
            facts.setdefault(row["metadata"]["fact_id"], []).append(row)
            self.assertEqual(row["schema_version"], "position-group-v2")
        self.assertEqual(len(facts), 2)
        for fact_rows in facts.values():
            self.assertEqual(
                len({row["metadata"]["fact_fingerprint"] for row in fact_rows}), 1
            )
            self.assertEqual(len({str(row["target"]) for row in fact_rows}), 1)
            replica_fillers = {
                (row["metadata"]["replica_index"], row["metadata"]["filler_fingerprint"])
                for row in fact_rows
            }
            self.assertEqual(len({replica for replica, _ in replica_fillers}), 4)
            self.assertEqual(len({fingerprint for _, fingerprint in replica_fillers}), 4)

    def test_matched_variants_hold_facts_replicas_and_fillers_constant(self) -> None:
        bank = list(
            iter_matched_training_bank(
                split="train",
                facts_per_condition=4,
                replicas_per_fact=4,
                tasks=("kv",),
                filler_types=("neutral",),
                target_lengths=(512,),
                positions=(0.0, 0.25, 0.5, 1.0),
                seed=23,
                token_counter=self.counter,
                words_per_document=16,
            )
        )
        variants = prepare_matched.build_variants(
            bank,
            seed=23,
            max_token_budget_gap=0.01,
        )
        self.assertEqual(set(variants), set(prepare_matched.VARIANTS))
        self.assertEqual({len(rows) for rows in variants.values()}, {16})

        paired = variants[("paired_answer")]
        independent = variants[("independent_answer")]
        for rows, expected_position_count in ((paired, 4), (independent, 1)):
            per_fact: dict[str, set[str]] = {}
            for row in rows:
                metadata = row["metadata"]
                per_fact.setdefault(metadata["design_fact_id"], set()).add(
                    metadata["position_label"]
                )
            self.assertEqual({len(values) for values in per_fact.values()}, {expected_position_count})

        def matched_cells(rows):
            return {
                (row["metadata"]["design_fact_id"], row["metadata"]["replica_index"]): (
                    row["metadata"]["fact_fingerprint"],
                    row["metadata"]["filler_fingerprint"],
                    row["metadata"]["actual_tokens"],
                )
                for row in rows
            }

        self.assertEqual(matched_cells(paired), matched_cells(independent))
        summary = prepare_matched.summarize(variants)
        self.assertEqual(summary["training_design"], "matched-position-v1")
        with tempfile.TemporaryDirectory() as temporary_dir:
            paths = []
            for name, rows in variants.items():
                path = Path(temporary_dir) / f"{name}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        for row in rows
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            audit = audit_matched.audit(paths, max_token_budget_gap=0.01)
        self.assertEqual(audit["status"], "ok")
        self.assertTrue(audit["pairing_comparison"]["same_filler_views"])

        expected_payload_shapes = {
            "answer": (False, False),
            "evidence_id": (True, False),
            "evidence": (True, True),
        }
        for name, rows in variants.items():
            supervision = rows[0]["metadata"]["supervision_mode"]
            payload = json.loads(rows[0]["messages"][-1]["content"])
            self.assertEqual(
                (bool(payload["evidence_ids"]), bool(payload["evidence_quotes"])),
                expected_payload_shapes[supervision],
                name,
            )

    def test_diagnostic_modes_deduplicate_short_oracle(self) -> None:
        source = generate_position_equivalent_group(
            split="test",
            group_index=0,
            task="two_hop",
            filler_type="same_format",
            target_tokens=512,
            positions=(0.0, 0.25, 0.5, 1.0),
            seed=31,
            token_counter=self.counter,
            words_per_document=16,
        )
        rows = list(
            prepare_diagnostics.derive_rows(
                source,
                list(prepare_diagnostics.SUPPORTED_MODES),
                token_counter=self.counter,
            )
        )
        self.assertEqual(len(rows), 9)
        counts = {
            mode: sum(row["evaluation_mode"] == mode for row in rows)
            for mode in prepare_diagnostics.SUPPORTED_MODES
        }
        self.assertEqual(
            counts,
            {"locate_only": 4, "oracle_long": 4, "oracle_short": 1},
        )
        oracle_short = [row for row in rows if row["evaluation_mode"] == "oracle_short"][0]
        oracle_long = [row for row in rows if row["evaluation_mode"] == "oracle_long"][0]
        self.assertEqual(oracle_short["position_label"], "oracle")
        self.assertNotIn("filler-", oracle_short["prompt"])
        for quote in oracle_short["target"]["evidence_quotes"]:
            self.assertIn(quote, oracle_short["prompt"])
        for evidence_id in oracle_long["target"]["evidence_ids"]:
            self.assertEqual(oracle_long["prompt"].count(f"[Document {evidence_id}]"), 1)
        self.assertTrue(
            oracle_long["metadata"]["oracle_evidence_moved_from_source_context"]
        )


if __name__ == "__main__":
    unittest.main()
