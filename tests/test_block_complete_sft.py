import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_block_complete_sft",
    ROOT / "scripts" / "materialize_block_complete_sft.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_row(task: str, position: str, fact_index: int, replica: int) -> dict:
    fact = f"{task}-{position}-{fact_index}"
    return {
        "schema_version": "position-sft-v2",
        "messages": [],
        "metadata": {
            "pairing_mode": "independent",
            "design_fact_id": fact,
            "replica_index": replica,
            "task": task,
            "filler_type": "neutral",
            "target_tokens": 8192,
            "assigned_position": position,
        },
    }


class BlockCompleteSelectionTests(unittest.TestCase):
    def test_materializer_has_explicit_native_protocol_fallback(self) -> None:
        source = (ROOT / "scripts" / "materialize_block_complete_sft.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('parent_metadata.get("chat_protocol") or NATIVE', source)

    def setUp(self) -> None:
        self.rows = []
        for task in ("kv", "two_hop"):
            for position in ("p000", "p025", "p050", "p100"):
                for fact_index in range(5):
                    for replica in range(4):
                        self.rows.append(
                            make_row(task, position, fact_index, replica)
                        )

    def test_selection_is_96_rows_of_complete_balanced_blocks(self) -> None:
        indices, report = MODULE.select_block_indices(
            self.rows, seed=20260825, facts_per_stratum=3
        )
        selected = [self.rows[index] for index in indices]
        self.assertEqual(len(indices), 96)
        self.assertEqual(report["selected_facts"], 24)
        self.assertEqual(report["replicas_per_fact"], 4)
        self.assertEqual(report["stratum_count"], 8)
        facts = {}
        for row in selected:
            facts.setdefault(row["metadata"]["design_fact_id"], set()).add(
                row["metadata"]["replica_index"]
            )
        self.assertEqual(set(map(len, facts.values())), {4})
        counts = {}
        for row in selected:
            key = (row["metadata"]["task"], row["metadata"]["assigned_position"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(set(counts.values()), {12})

    def test_selection_is_deterministic_and_seeded(self) -> None:
        left, left_report = MODULE.select_block_indices(
            self.rows, seed=1, facts_per_stratum=3
        )
        same, same_report = MODULE.select_block_indices(
            self.rows, seed=1, facts_per_stratum=3
        )
        other, _ = MODULE.select_block_indices(
            self.rows, seed=2, facts_per_stratum=3
        )
        self.assertEqual(left, same)
        self.assertEqual(
            left_report["selected_indices_sha256"],
            same_report["selected_indices_sha256"],
        )
        self.assertNotEqual(left, other)

    def test_incomplete_stratum_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs 6"):
            MODULE.select_block_indices(
                self.rows, seed=1, facts_per_stratum=6
            )

    def test_existing_materialization_is_reused_only_when_lineage_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "manifest_sha256": root / "manifest.json",
                "matched_audit_sha256": root / "matched-audit.json",
                "selection_sha256": root / "selection.json",
            }
            for index, path in enumerate(files.values()):
                path.write_text(json.dumps({"value": index}) + "\n", encoding="utf-8")
            completion = {
                "schema_version": "block-complete-sft-seed-v1",
                "status": "validated",
                "seed": 7,
                "rows_per_variant": 96,
                "optimizer_steps": 96,
                "facts_per_stratum": 3,
                "positions_per_variant": {
                    "p000": 24,
                    "p025": 24,
                    "p050": 24,
                    "p100": 24,
                },
                "strict_realized_matching": True,
                "complete_fact_blocks": True,
                **{field: MODULE.sha256_file(path) for field, path in files.items()},
            }
            MODULE.validate_existing_materialization(
                root, completion, seed=7, facts_per_stratum=3, expected_rows=96
            )
            files["selection_sha256"].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Existing materialized lineage hash mismatch"
            ):
                MODULE.validate_existing_materialization(
                    root, completion, seed=7, facts_per_stratum=3, expected_rows=96
                )


if __name__ == "__main__":
    unittest.main()
