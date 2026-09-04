from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "stage_model.py"


class StageModelAttestationTest(unittest.TestCase):
    def test_manifest_only_writes_nonrecursive_integrity_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            actual = {
                "model_type": "fake",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "max_position_embeddings": 128,
            }
            (model / "config.json").write_text(json.dumps(actual), encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "chat_protocol_audit.json").write_text("{}\n", encoding="utf-8")
            config = root / "experiment.json"
            config.write_text(
                json.dumps(
                    {
                        "model_id": "test/fake",
                        "revision": "a" * 40,
                        "config_signature": actual,
                    }
                ),
                encoding="utf-8",
            )
            for _ in range(2):
                subprocess.run(
                    [
                        "python3",
                        str(SCRIPT),
                        "--config",
                        str(config),
                        "--output",
                        str(model),
                        "--manifest-only",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            manifest_path = model / "model_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            names = {item["path"] for item in manifest["files"]}
            self.assertNotIn("model_integrity_attestation.json", names)
            self.assertNotIn("chat_protocol_audit.json", names)
            attestation = json.loads(
                (model / "model_integrity_attestation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                attestation["manifest_sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(len(attestation["file_state"]), len(manifest["files"]))


if __name__ == "__main__":
    unittest.main()
