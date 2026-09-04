import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_full_paper_completion.py"
SPEC = importlib.util.spec_from_file_location("audit_full_paper_completion", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class FullPaperCompletionAuditTest(unittest.TestCase):
    def test_validates_package_checksum_and_project_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "suite.tar.gz"
            artifact.write_bytes(b"package")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            artifact.with_suffix(".gz.sha256").write_text(
                f"{digest}  {artifact.name}\n", encoding="utf-8"
            )
            evidence = root / "results.json"
            evidence.write_text(
                json.dumps(
                    {
                        "schema_version": "active-gpu-compute-accounting-v1",
                        "status": "validated",
                        "expected_training_step": 96,
                        "hourly_rate_cny": 2.78,
                        "by_kind": {
                            "training": {"events": 1},
                            "evaluation": {"events": 1},
                        },
                        "total_active_gpu_seconds": 2.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "manifest.json"
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--project-root",
                    str(root),
                    "--artifact",
                    f"suite={artifact}",
                    "--evidence",
                    f"compute_accounting={evidence}",
                    "--require-evidence-label",
                    "compute_accounting",
                    "--output",
                    str(output),
                ]
                self.assertEqual(audit.main(), 0)
            finally:
                sys.argv = old_argv
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "validated")
            self.assertEqual(manifest["artifact_count"], 1)
            self.assertEqual(
                manifest["required_evidence_labels"], ["compute_accounting"]
            )
            self.assertTrue(manifest["final_release_ready"])
            self.assertEqual(
                manifest["evidence"]["compute_accounting"]["path"], "results.json"
            )

    def test_rejects_missing_required_evidence_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "suite.tar.gz"
            artifact.write_bytes(b"package")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            artifact.with_suffix(".gz.sha256").write_text(
                f"{digest}  {artifact.name}\n", encoding="utf-8"
            )
            evidence = root / "results.json"
            evidence.write_text('{"status":"validated"}\n', encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--project-root",
                    str(root),
                    "--artifact",
                    f"suite={artifact}",
                    "--evidence",
                    f"result={evidence}",
                    "--require-evidence-label",
                    "compute_accounting",
                    "--output",
                    str(root / "manifest.json"),
                ]
                with self.assertRaisesRegex(SystemExit, "compute_accounting"):
                    audit.main()
            finally:
                sys.argv = old_argv

    def test_rejects_bad_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "suite.tar.gz"
            artifact.write_bytes(b"package")
            artifact.with_suffix(".gz.sha256").write_text(
                "0" * 64 + "  suite.tar.gz\n", encoding="utf-8"
            )
            with self.assertRaises(SystemExit):
                audit.audit_artifact("suite", artifact)

    def test_rejects_empty_or_wrong_step_compute_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compute.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "active-gpu-compute-accounting-v1",
                        "status": "validated",
                        "expected_training_step": 100,
                        "hourly_rate_cny": 2.78,
                        "by_kind": {
                            "training": {"events": 0},
                            "evaluation": {"events": 0},
                        },
                        "total_active_gpu_seconds": 0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "checkpoint-96"):
                audit.validate_evidence_semantics("compute_accounting", path)

    def test_rejects_cross_family_confirmatory_only_relabeling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "seed-level-analysis-v1",
                        "analysis_kind": "factorial",
                        "primary_training_seed_summary": True,
                        "confirmatory_only_primary_summary": True,
                        "primary_statuses_by_family": audit.EXPECTED_PRIMARY_STATUSES,
                        "families": {
                            family: {} for family in audit.EXPECTED_PRIMARY_STATUSES
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "designation"):
                audit.validate_evidence_semantics("cross_family_rule", path)

    def test_rejects_paper_tex_not_bound_to_generation_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {}
            sources = {}
            for source_name, evidence_label in audit.PAPER_RESULT_SOURCE_LABELS.items():
                path = root / f"{evidence_label}.json"
                path.write_text('{"status":"validated"}\n', encoding="utf-8")
                entries[evidence_label] = path
                sources[source_name] = {
                    "path": str(path.resolve()),
                    "sha256": audit.sha256_file(path),
                }
            exploratory = root / "exploratory.json"
            exploratory.write_text('{"status":"validated"}\n', encoding="utf-8")
            sources["qwen_exploratory_rule"] = {
                "path": str(exploratory.resolve()),
                "sha256": audit.sha256_file(exploratory),
            }
            tex = root / "results.tex"
            tex.write_text("generated\n", encoding="utf-8")
            entries["paper_results_tex"] = tex
            manifest = root / "results.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "sources": sources,
                        "output_tex_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            entries["paper_results_manifest"] = manifest
            with self.assertRaisesRegex(SystemExit, "TeX changed"):
                audit.validate_derived_manifest_bindings(entries, root.resolve())


if __name__ == "__main__":
    unittest.main()
