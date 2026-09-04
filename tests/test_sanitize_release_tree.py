from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "sanitize_release_tree.py"
SPEC = importlib.util.spec_from_file_location("sanitize_release_tree", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SanitizeReleaseTreeTest(unittest.TestCase):
    def test_remote_paths_are_portable_and_hash_lineage_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw"
            source.mkdir()
            remote_root = "/" + "root/autodl-tmp"
            (source / "run.json").write_text(
                json.dumps(
                    {
                        "model": remote_root + "/models/Mistral-7B-Instruct-v0.3",
                        "data": remote_root + "/position-bias-pilot/data/test.jsonl",
                    }
                ),
                encoding="utf-8",
            )
            output = Path(directory) / "release"
            subprocess.run(
                ["python3", str(SCRIPT), "--input", str(source), "--output", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            text = (output / "run.json").read_text(encoding="utf-8")
            self.assertNotIn("/root/", text)
            self.assertIn("models/Mistral-7B-Instruct-v0.3", text)
            manifest = json.loads(
                (output / "sanitization_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "validated")
            self.assertEqual(manifest["total_path_replacements"], 2)

    def test_credential_pattern_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "credential"):
            MODULE.sanitize_text("pass" + "word: secret")


if __name__ == "__main__":
    unittest.main()
