from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_nolima_ood.py"
SPEC = importlib.util.spec_from_file_location("prepare_nolima_ood", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
nolima = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nolima
SPEC.loader.exec_module(nolima)


class CharacterTokenizer:
    name_or_path = "character-tokenizer"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, values, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(value) for value in values)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered) if tokenize else rendered


class MappingCharacterTokenizer(CharacterTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        values = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        return {"input_ids": values, "attention_mask": [1] * len(values)}


class NoLiMaOODTest(unittest.TestCase):
    def test_release_paths_do_not_capture_machine_roots(self) -> None:
        self.assertEqual(
            nolima.portable_path(ROOT / "data" / "fixture.jsonl"),
            "data/fixture.jsonl",
        )
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = Path(directory) / "local-model"
            tokenizer.mkdir()
            self.assertEqual(nolima.portable_tokenizer_name(str(tokenizer)), "local-model")

    def test_chat_token_count_handles_mapping_outputs(self) -> None:
        tokenizer = MappingCharacterTokenizer()
        count = nolima.chat_token_count(tokenizer, "system", "user")
        self.assertGreater(count, 10)

    def test_expand_and_generate_matched_position_groups(self) -> None:
        needle_set = [
            {
                "id": "case",
                "reasoning_type": "world_knowledge",
                "needle": "Actually, {CHAR} lives next to {1}.",
                "questions": {"twohop": "Who has been to {2}?"},
                "character_set": ["Yuki", "Stuart"],
                "tests": {
                    "test": {"input_args": ["the museum", "the capital"]}
                },
            }
        ]
        cases = nolima.expand_cases(needle_set)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].question, "Who has been to the capital?")
        self.assertEqual(
            cases[0].needle_template, "Actually, {CHAR} lives next to the museum."
        )

        with tempfile.TemporaryDirectory() as directory:
            book = Path(directory) / "book.txt"
            book.write_text(
                "\n".join(f"Line {index}: unrelated archive text." for index in range(200)),
                encoding="utf-8",
            )
            rows = list(
                nolima.generate_rows(
                    tokenizer=CharacterTokenizer(),
                    cases=cases,
                    books=[book],
                    lengths=[700, 900],
                    positions=[0.0, 0.5, 1.0],
                    seed=17,
                    needle_set_name="fixture",
                )
            )

        self.assertEqual(len(rows), 6)
        audit = nolima.audit_rows(rows, positions=[0.0, 0.5, 1.0])
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["groups"], 2)
        self.assertEqual(audit["positions_per_group"], 3)
        groups = {}
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
            self.assertEqual(row["target"]["evidence_ids"], [])
            self.assertIn(row["target"]["evidence_quotes"][0], row["prompt"])
            self.assertLessEqual(row["actual_tokens"], row["target_tokens"])
        for group in groups.values():
            self.assertEqual(len({json.dumps(row["target"], sort_keys=True) for row in group}), 1)
            self.assertEqual(
                len({row["metadata"]["base_context_sha256"] for row in group}), 1
            )
            self.assertEqual(
                {row["position_label"] for row in group}, {"p000", "p050", "p100"}
            )


if __name__ == "__main__":
    unittest.main()
