from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "preflight_autodl.py"
SPEC = importlib.util.spec_from_file_location("preflight_autodl", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ModelIntegrityAttestationTest(unittest.TestCase):
    def test_legacy_missing_or_null_chat_protocol_means_native(self):
        native = "native-system-user-assistant"
        self.assertEqual(MODULE.metadata_chat_protocol({}), native)
        self.assertEqual(MODULE.metadata_chat_protocol({"chat_protocol": None}), native)
        self.assertEqual(
            MODULE.metadata_chat_protocol({"chat_protocol": "merge-system-into-first-user-v1"}),
            "merge-system-into-first-user-v1",
        )

    def make_model(self, root: Path) -> Path:
        model = root / "model"
        model.mkdir()
        for name, content in {
            "config.json": "{}\n",
            "tokenizer_config.json": "{}\n",
            "model.safetensors": "weights\n",
        }.items():
            (model / name).write_text(content, encoding="utf-8")
        files = []
        for path in sorted(model.iterdir()):
            files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": MODULE.sha256_file(path),
                }
            )
        (model / "model_manifest.json").write_text(
            json.dumps({"revision": "a" * 40, "files": files}) + "\n",
            encoding="utf-8",
        )
        return model

    def test_full_hash_then_cached_stat_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(Path(directory))
            report, attestation = MODULE.verify_model(model)
            self.assertEqual(report["verification_mode"], "full-sha256")
            self.assertIsNotNone(attestation)
            attestation_path = Path(directory) / "attestation.json"
            attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
            cached_report, _ = MODULE.verify_model(model, attestation_path)
            self.assertEqual(
                cached_report["verification_mode"],
                "cached-sha256-with-stat-revalidation",
            )

            with (model / "model.safetensors").open("a", encoding="utf-8") as handle:
                handle.write("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE.verify_model(model, attestation_path)

    def test_model_contract_binds_config_tokenizer_protocol_and_audit(self):
        class Backend:
            @staticmethod
            def to_str():
                return "backend-v1"

        class Tokenizer:
            backend_tokenizer = Backend()
            chat_template = "template-v1"
            special_tokens_map = {"bos_token": "<s>"}

            def __init__(self, name_or_path: str):
                self.name_or_path = name_or_path

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, **_kwargs):
                return Tokenizer(str(path))

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps({"model_type": "fake", "hidden_size": 8}),
                encoding="utf-8",
            )
            audit = {
                "status": "passed",
                "selected_protocol": "merge-system-into-first-user-v1",
            }
            audit_path = model / "chat_protocol_audit.json"
            audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
            tokenizer = Tokenizer(str(model))
            contract = {
                "config_signature": {"model_type": "fake", "hidden_size": 8},
                "tokenizer_fingerprint": MODULE.tokenizer_fingerprint(tokenizer),
                "chat_protocol": "merge-system-into-first-user-v1",
                "chat_protocol_audit_sha256": hashlib.sha256(
                    audit_path.read_bytes()
                ).hexdigest(),
            }
            fake_transformers = types.SimpleNamespace(AutoTokenizer=AutoTokenizer)
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                report = MODULE.verify_model_contract(model, contract)
            self.assertTrue(report["config_signature_verified"])
            self.assertEqual(report["chat_protocol"], contract["chat_protocol"])

            contract["chat_protocol_audit_sha256"] = "0" * 64
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                with self.assertRaisesRegex(ValueError, "audit differs"):
                    MODULE.verify_model_contract(model, contract)


if __name__ == "__main__":
    unittest.main()
