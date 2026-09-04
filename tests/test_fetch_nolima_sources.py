from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "fetch_nolima_sources.py"
SPEC = importlib.util.spec_from_file_location("fetch_nolima_sources", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FetchNoLiMaSourcesTest(unittest.TestCase):
    def test_reconstructs_upstream_same_name_continuation(self):
        self.assertEqual(
            MODULE.reconstruct_continued_book(b"normal", b"long-file-tail"),
            b"normalile-tail",
        )

    def test_rejects_nonlong_source(self):
        with self.assertRaises(ValueError):
            MODULE.reconstruct_continued_book(b"normal", b"short")

    def test_frozen_combined_hashes_match_qwen_manifest(self):
        self.assertEqual(
            [MODULE.BOOKS[index]["combined"] for index in sorted(MODULE.BOOKS)],
            [
                "290e84ffbb59b0a7af1a01b200d808692ee99018c220dace8b6d333cdac68cfe",
                "bc3a5245f0d556bb6d24e4064649c31267978529b58e6165d3a1d38191c33363",
                "6eac8fcb2bcd4bfc9c6298008463ba565c65485816a6f0eeb73cc7938ca7fa11",
                "34cdad0fa7362ba270c4afc39f5acb3d65c39b00eefdf37b426cd0ac53f1a9b6",
                "aeff17579398c5c995907d772b76783b93ec5f21ace4cf186f2f856e90aa10f8",
            ],
        )


if __name__ == "__main__":
    unittest.main()
