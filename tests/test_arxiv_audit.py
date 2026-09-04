from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_arxiv_source.py"
SPEC = importlib.util.spec_from_file_location("audit_arxiv_source", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class ArxivAuditTest(unittest.TestCase):
    def make_tree(self, root: Path, pending: bool = False) -> None:
        (root / "generated").mkdir(parents=True)
        (root / "main.tex").write_text(
            "\\documentclass{article}\\begin{document}"
            + ("[PENDING: result] " if pending else "")
            + "Evidence \\citep{paper}.\\end{document}",
            encoding="utf-8",
        )
        (root / "references.bib").write_text(
            "@article{paper, title={Test}, author={A}, year={2026}}\n",
            encoding="utf-8",
        )
        (root / "generated/results.tex").write_text("% generated\n", encoding="utf-8")
        (root / "main.bbl").write_text("% compiled bibliography\n", encoding="utf-8")
        (root / "README.md").write_text("Build instructions.\n", encoding="utf-8")

    def invoke(self, root: Path, output: Path, allow_pending: bool = False) -> int:
        old_argv = sys.argv
        try:
            sys.argv = [str(SCRIPT), "--paper-dir", str(root), "--output", str(output)]
            if allow_pending:
                sys.argv.append("--allow-pending")
            return audit.main()
        finally:
            sys.argv = old_argv

    def test_clean_source_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root)
            output = root / "audit.json"
            self.assertEqual(self.invoke(root, output), 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["status"], "passed")
            self.assertIn("main.bbl", payload["files"])
            self.assertEqual(
                payload["files"]["main.bbl"]["bytes"],
                (root / "main.bbl").stat().st_size,
            )

    def test_pending_fails_submission_but_passes_scaffold_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root, pending=True)
            output = root / "audit.json"
            self.assertEqual(self.invoke(root, output), 1)
            self.assertEqual(self.invoke(root, output, allow_pending=True), 0)
            self.assertTrue(json.loads(output.read_text())["pending"])

    def test_undefined_cross_reference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root)
            path = root / "main.tex"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "Evidence", r"See Figure~\ref{fig:missing}. Evidence"
                ),
                encoding="utf-8",
            )
            output = root / "audit.json"
            self.assertEqual(self.invoke(root, output), 1)
            codes = {item["code"] for item in json.loads(output.read_text())["errors"]}
            self.assertIn("undefined_reference", codes)

    def test_documentation_can_name_pending_guard_without_becoming_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root)
            (root / "README.md").write_text(
                "The manuscript uses uppercase PENDING guards while work is incomplete.\n",
                encoding="utf-8",
            )
            output = root / "audit.json"
            self.assertEqual(self.invoke(root, output), 0)
            self.assertFalse(json.loads(output.read_text())["pending"])


if __name__ == "__main__":
    unittest.main()
