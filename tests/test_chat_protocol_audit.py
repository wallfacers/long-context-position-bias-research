from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_chat_protocol.py"
SPEC = importlib.util.spec_from_file_location("audit_chat_protocol", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    chat_template = "fake-v1"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = "|".join(f"{row['role']}:{row['content']}" for row in messages)
        if add_generation_prompt:
            rendered += "|assistant:"
        if tokenize:
            return [ord(character) for character in rendered]
        return rendered


class MistralLikeTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        filtered = [row for row in messages if row["role"] != "system"]
        return super().apply_chat_template(
            filtered, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )


class ChatProtocolAuditTest(unittest.TestCase):
    def test_native_protocol_and_completion_prefix(self):
        result = MODULE.audit_tokenizer(FakeTokenizer())
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["completion_mask_prefix_safe"])
        self.assertEqual(result["completion_tokens"], len("ASSISTANT_SENTINEL_98ab"))

    def test_native_system_bug_selects_explicit_merge_protocol(self):
        result = MODULE.audit_tokenizer(MistralLikeTokenizer())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["selected_protocol"], "merge-system-into-first-user-v1")
        self.assertEqual(result["native_protocol_status"], "failed")


if __name__ == "__main__":
    unittest.main()
