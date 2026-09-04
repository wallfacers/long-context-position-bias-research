import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_public_release.py"
SPEC = importlib.util.spec_from_file_location("build_public_release", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(release)


class BuildPublicReleaseTest(unittest.TestCase):
    @staticmethod
    def final_evidence(root: Path, evidence: Path) -> Path:
        compute = root / "results/compute_accounting.json"
        compute.parent.mkdir(parents=True, exist_ok=True)
        compute.write_text('{"status":"validated"}\n', encoding="utf-8")
        record = {
            "path": "results/compute_accounting.json",
            "bytes": compute.stat().st_size,
            "sha256": release.sha256_file(compute),
        }
        evidence.write_text(
            json.dumps(
                {
                    "schema_version": "full-paper-evidence-manifest-v1",
                    "status": "validated",
                    "required_evidence_labels": ["compute_accounting"],
                    "evidence": {"compute_accounting": record},
                    "final_release_ready": True,
                }
            ),
            encoding="utf-8",
        )
        return compute

    def test_excludes_raw_jsonl_weights_and_keeps_audited_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "project"
            root.mkdir()
            (root / "README.md").write_text("project\n", encoding="utf-8")
            (root / "scripts").mkdir()
            (root / "scripts/run.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "data/benchmark").mkdir(parents=True)
            (root / "data/benchmark/raw.jsonl").write_text('{"prompt":"licensed"}\n')
            (root / "data/benchmark/manifest.json").write_text('{"status":"ok"}\n')
            (root / "results/suite").mkdir(parents=True)
            (root / "results/suite/base.jsonl").write_text('{"generated":"text"}\n')
            (root / "results/suite/analysis.json").write_text('{"score":0.5}\n')
            (root / "outputs/run/checkpoint-100").mkdir(parents=True)
            (root / "outputs/run/checkpoint-100/adapter_model.safetensors").write_bytes(b"x")
            (root / "third_party/nltk_data").mkdir(parents=True)
            (root / "third_party/nltk_data/punkt.sha256").write_text(
                "0" * 64 + "  english/abbrev_types.txt\n", encoding="utf-8"
            )
            (root / "third_party/nltk_data/english").mkdir()
            (root / "third_party/nltk_data/english/abbrev_types.txt").write_text(
                "licensed data\n", encoding="utf-8"
            )
            (root / "third_party/NoLiMa/data").mkdir(parents=True)
            (root / "third_party/NoLiMa/data/frozen-source-download-manifest.json").write_text(
                '{"status":"validated","dataset_revision":"frozen"}\n',
                encoding="utf-8",
            )
            evidence = root / "results/full_paper_evidence_manifest.json"
            self.final_evidence(root, evidence)
            output = parent / "public"
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--project-root",
                    str(root),
                    "--evidence-manifest",
                    str(evidence),
                    "--output",
                    str(output),
                ]
                self.assertEqual(release.main(), 0)
            finally:
                sys.argv = old_argv
            self.assertTrue((output / "results/suite/analysis.json").is_file())
            self.assertTrue((output / "data/benchmark/manifest.json").is_file())
            self.assertFalse((output / "data/benchmark/raw.jsonl").exists())
            self.assertFalse((output / "results/suite/base.jsonl").exists())
            self.assertFalse(
                (output / "outputs/run/checkpoint-100/adapter_model.safetensors").exists()
            )
            self.assertTrue((output / "third_party/nltk_data/punkt.sha256").is_file())
            self.assertTrue(
                (
                    output
                    / "third_party/NoLiMa/data/frozen-source-download-manifest.json"
                ).is_file()
            )
            self.assertFalse(
                (output / "third_party/nltk_data/english/abbrev_types.txt").exists()
            )
            policy = json.loads(
                (output / "public_release_selection.json").read_text(encoding="utf-8")
            )
            self.assertIn("Raw benchmark", policy["raw_jsonl_policy"])
            self.assertFalse(policy["evidence_completeness_preflight_bypass"])

    def test_rejects_manifest_without_compute_accounting_unless_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "project"
            root.mkdir()
            (root / "README.md").write_text("project\n", encoding="utf-8")
            evidence = root / "evidence.json"
            evidence.write_text(
                json.dumps(
                    {
                        "schema_version": "full-paper-evidence-manifest-v1",
                        "status": "validated",
                    }
                ),
                encoding="utf-8",
            )
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--project-root",
                    str(root),
                    "--evidence-manifest",
                    str(evidence),
                    "--output",
                    str(parent / "blocked"),
                ]
                with self.assertRaisesRegex(SystemExit, "compute_accounting"):
                    release.main()
                sys.argv.extend(["--preflight-allow-incomplete-evidence"])
                sys.argv[sys.argv.index(str(parent / "blocked"))] = str(parent / "preflight")
                self.assertEqual(release.main(), 0)
            finally:
                sys.argv = old_argv
            policy = json.loads(
                (parent / "preflight/public_release_selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(policy["evidence_completeness_preflight_bypass"])

    def test_ignores_excluded_environment_symlink_but_rejects_selected_source_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            (root / ".venv/bin").mkdir(parents=True)
            (root / "scripts").mkdir()
            target = root / ".venv/bin/python-real"
            target.write_text("binary placeholder\n", encoding="utf-8")
            (root / ".venv/bin/python").symlink_to(target.name)
            (root / "scripts/run.py").write_text("print('ok')\n", encoding="utf-8")

            selected, excluded = release.selected_files(root)
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in selected],
                ["scripts/run.py"],
            )
            self.assertGreaterEqual(excluded[".venv"], 1)

            (root / "scripts/link.py").symlink_to("run.py")
            with self.assertRaisesRegex(ValueError, "selected public source"):
                release.selected_files(root)

    def test_source_code_is_secret_scanned_not_copied_as_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "project"
            (root / "scripts").mkdir(parents=True)
            secret_fixture = "pass" + 'word = "must-not-publish"\n'
            (root / "scripts/leak.py").write_text(secret_fixture, encoding="utf-8")
            (root / "results").mkdir()
            evidence = root / "results/full_paper_evidence_manifest.json"
            self.final_evidence(root, evidence)
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--project-root",
                    str(root),
                    "--evidence-manifest",
                    str(evidence),
                    "--output",
                    str(parent / "public"),
                ]
                with self.assertRaises(ValueError):
                    release.main()
            finally:
                sys.argv = old_argv

    def test_rejects_evidence_changed_after_final_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "project"
            root.mkdir()
            (root / "README.md").write_text("project\n", encoding="utf-8")
            evidence = root / "results/full_paper_evidence_manifest.json"
            compute = self.final_evidence(root, evidence)
            compute.write_text('{"status":"tampered"}\n', encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--project-root",
                    str(root),
                    "--evidence-manifest",
                    str(evidence),
                    "--output",
                    str(parent / "public"),
                ]
                with self.assertRaisesRegex(
                    SystemExit, "changed after final audit: compute_accounting"
                ):
                    release.main()
            finally:
                sys.argv = old_argv
            self.assertFalse((parent / "public").exists())

    def test_rejects_final_manifest_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "project"
            root.mkdir()
            evidence = parent / "outside-evidence.json"
            self.final_evidence(root, evidence)
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--project-root",
                    str(root),
                    "--evidence-manifest",
                    str(evidence),
                    "--output",
                    str(parent / "public"),
                ]
                with self.assertRaisesRegex(SystemExit, "inside the project tree"):
                    release.main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
