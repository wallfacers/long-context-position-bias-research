import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_realized_training_subset",
    ROOT / "scripts" / "audit_realized_training_subset.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(fact: str, replica: int, position: str, variant: str) -> dict:
    pairing, supervision = variant.split("_", maxsplit=1)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": f"{fact}:{replica}:{position}"},
        {"role": "assistant", "content": supervision},
    ]
    return {
        "schema_version": "position-sft-v2",
        "id": f"{variant}:{fact}:{replica}",
        "messages": messages,
        "metadata": {
            "variant": variant,
            "design_fact_id": fact,
            "replica_index": replica,
            "raw_sample_id": f"raw:{fact}:{replica}:{position}",
            "position_label": position,
            "fact_fingerprint": f"fact-hash:{fact}",
            "filler_fingerprint": f"fill-hash:{fact}:{replica}",
            "actual_tokens": 10,
            "task": "kv",
            "filler_type": "neutral",
            "target_tokens": 100,
            "pairing_mode": pairing,
            "supervision_mode": supervision,
        },
    }


def tokenized(length: int = 8, completion: int = 2) -> dict:
    return {
        "input_ids": list(range(length)),
        "completion_mask": [0] * (length - completion) + [1] * completion,
    }


class RealizedSubsetAuditTests(unittest.TestCase):
    def test_legacy_null_protocol_is_explicitly_audited_as_native(self) -> None:
        native = "native-system-user-assistant"
        self.assertEqual(MODULE.normalized_chat_protocol({}), native)
        self.assertEqual(MODULE.normalized_chat_protocol({"chat_protocol": None}), native)
        self.assertEqual(
            MODULE.normalized_chat_protocol(
                {"chat_protocol": "merge-system-into-first-user-v1"}
            ),
            "merge-system-into-first-user-v1",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            MODULE.normalized_chat_protocol({"chat_protocol": "unknown"})

    def test_exposure_statistics_distinguish_actual_cross_position_pairing(self) -> None:
        independent = [
            row("f1", 0, "p000", "independent_answer"),
            row("f1", 1, "p000", "independent_answer"),
            row("f2", 0, "p025", "independent_answer"),
            row("f3", 0, "p050", "independent_answer"),
        ]
        paired = [
            row("f1", 0, "p000", "paired_answer"),
            row("f1", 1, "p025", "paired_answer"),
            row("f2", 0, "p100", "paired_answer"),
            row("f3", 0, "p050", "paired_answer"),
        ]
        tokens = [tokenized() for _ in independent]
        indices = list(range(4))
        summaries = {
            "independent_answer": MODULE.summarize_selected_rows(
                independent, tokens, indices
            ),
            "paired_answer": MODULE.summarize_selected_rows(paired, tokens, indices),
        }
        comparison = MODULE.pairing_comparison(
            independent,
            paired,
            indices,
            summaries,
            max_prompt_token_gap=0.002,
        )
        self.assertEqual(
            summaries["independent_answer"]["facts_with_cross_position_exposure"], 0
        )
        self.assertEqual(
            summaries["paired_answer"]["facts_with_cross_position_exposure"], 1
        )
        self.assertTrue(comparison["same_ordered_fact_replica_identities"])
        self.assertFalse(comparison["position_histograms_exactly_equal"])
        self.assertFalse(comparison["paired_blocks_complete_for_every_selected_fact"])
        self.assertFalse(comparison["strict_realized_fixed_step_matching"])

    def test_supervision_siblings_must_keep_the_prompt_identical(self) -> None:
        rows_by_variant = {}
        for pairing in MODULE.PAIRING_MODES:
            for supervision in MODULE.SUPERVISION_MODES:
                variant = f"{pairing}_{supervision}"
                rows_by_variant[variant] = [row("f1", 0, "p000", variant)]
        rows_by_variant["paired_evidence"][0]["messages"][1]["content"] = "changed"
        with self.assertRaisesRegex(ValueError, "changes prompt"):
            MODULE.validate_supervision_siblings(rows_by_variant, [0], "paired")

    def test_completion_mask_must_be_one_suffix(self) -> None:
        rows = [row("f1", 0, "p000", "paired_answer")]
        malformed = [{"input_ids": [1, 2, 3], "completion_mask": [0, 1, 0]}]
        with self.assertRaisesRegex(ValueError, "single suffix"):
            MODULE.summarize_selected_rows(rows, malformed, [0])


if __name__ == "__main__":
    unittest.main()
