from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_pinned_model_config.py"
SPEC = importlib.util.spec_from_file_location("build_pinned_model_config", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PinnedModelConfigTest(unittest.TestCase):
    def test_signature_is_exact_and_rejects_missing_architecture(self):
        payload = {key: index for index, key in enumerate(MODULE.SIGNATURE_KEYS)}
        self.assertEqual(MODULE.config_signature(payload), payload)
        del payload["hidden_size"]
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            MODULE.config_signature(payload)


if __name__ == "__main__":
    unittest.main()
