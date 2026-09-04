from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "package_arxiv_source.py"


class ArxivPackageTest(unittest.TestCase):
    def test_deterministic_package_requires_and_contains_bbl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "paper"
            (root / "generated").mkdir(parents=True)
            (root / "main.tex").write_text(
                r"\documentclass{article}\begin{document}Evidence \citep{paper}.\end{document}",
                encoding="utf-8",
            )
            (root / "references.bib").write_text(
                "@article{paper,title={Test},author={A},year={2026}}\n",
                encoding="utf-8",
            )
            (root / "main.bbl").write_text("% bbl\n", encoding="utf-8")
            (root / "generated/results.tex").write_text("% numbers\n", encoding="utf-8")
            (root / "README.md").write_text("Build.\n", encoding="utf-8")
            evidence = Path(directory) / "full_paper_evidence_manifest.json"
            compute = Path(directory) / "compute.json"
            compute.write_text('{"status":"validated"}\n', encoding="utf-8")
            evidence.write_text(
                json.dumps(
                    {
                        "schema_version": "full-paper-evidence-manifest-v1",
                        "status": "validated",
                        "final_release_ready": True,
                        "required_evidence_labels": ["compute_accounting"],
                        "evidence": {
                            "compute_accounting": {
                                "path": "compute.json",
                                "bytes": compute.stat().st_size,
                                "sha256": hashlib.sha256(compute.read_bytes()).hexdigest(),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            hashes = []
            for index in range(2):
                archive = Path(directory) / f"source-{index}.tar.gz"
                manifest = Path(directory) / f"manifest-{index}.json"
                subprocess.run(
                    [
                        "python3",
                        str(SCRIPT),
                        "--paper-dir",
                        str(root),
                        "--evidence-manifest",
                        str(evidence),
                        "--output-tar",
                        str(archive),
                        "--output-manifest",
                        str(manifest),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertIn("main.bbl", {item["path"] for item in payload["files"]})
                self.assertTrue(
                    payload["full_evidence_manifest"]["final_release_ready"]
                )
                hashes.append(hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertEqual(hashes[0], hashes[1])

            compute.write_text("changed\n", encoding="utf-8")
            failed = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--paper-dir",
                    str(root),
                    "--evidence-manifest",
                    str(evidence),
                    "--output-tar",
                    str(Path(directory) / "tampered.tar.gz"),
                    "--output-manifest",
                    str(Path(directory) / "tampered.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("changed after final audit", failed.stderr + failed.stdout)


if __name__ == "__main__":
    unittest.main()
