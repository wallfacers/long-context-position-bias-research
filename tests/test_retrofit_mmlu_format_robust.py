from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "retrofit_mmlu_format_robust_analysis.sh"
RUN_FILES = {
    "base": "base.jsonl",
    "independent_answer": "independent_answer_s100.jsonl",
    "independent_evidence_id": "independent_evidence_id_s100.jsonl",
    "independent_evidence": "independent_evidence_s100.jsonl",
    "paired_answer": "paired_answer_s100.jsonl",
    "paired_evidence_id": "paired_evidence_id_s100.jsonl",
    "paired_evidence": "paired_evidence_s100.jsonl",
}


class RetrofitMmluFormatRobustTest(unittest.TestCase):
    def test_rescores_truncated_json_and_repacks_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            result = project / "results" / "mmlu"
            result.mkdir(parents=True)
            (project / "scripts").mkdir()
            shutil.copy2(
                ROOT / "scripts" / "analyze_general_regression.py",
                project / "scripts" / "analyze_general_regression.py",
            )
            (result / "RESULTS_READY_FOR_AGENT_REVIEW").touch()
            (result / "completion.json").write_text(
                json.dumps(
                    {
                        "schema_version": "mmlu-regression-completion-v1",
                        "status": "validated",
                        "paired_analysis": "general_regression_analysis/general_regression_analysis.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for run, filename in RUN_FILES.items():
                with (result / filename).open("w", encoding="utf-8") as handle:
                    for index in range(4):
                        handle.write(
                            json.dumps(
                                {
                                    "sample_id": f"mmlu/subject/{index}",
                                    "task": "mmlu_subject",
                                    "target": {"answer": "C"},
                                    "generated_text": '{"answer": "C", "confidence": 0.9',
                                    "parsed": None,
                                    "answer_score": 0,
                                    "valid_json": False,
                                    "finish_reason": "length",
                                }
                            )
                            + "\n"
                        )
            artifact = Path(temporary) / "mmlu.tar.gz"
            with tarfile.open(artifact, "w:gz") as archive:
                archive.add(result, arcname="results/mmlu")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            Path(str(artifact) + ".sha256").write_text(
                f"{digest}  {artifact}\n", encoding="utf-8"
            )

            subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--project-root",
                    str(project),
                    "--result-dir",
                    str(result),
                    "--artifact",
                    str(artifact),
                    "--expected-rows",
                    "4",
                    "--bootstrap-replicates",
                    "100",
                ],
                check=True,
                cwd=project,
                capture_output=True,
                text=True,
            )
            completion = json.loads((result / "completion.json").read_text())
            self.assertEqual(
                completion["scoring_protocol"], "format-robust-option-extraction-v1"
            )
            self.assertEqual(
                completion["legacy_format_constrained_analysis"],
                "general_regression_analysis/general_regression_analysis.json",
            )
            report = json.loads(
                (result / "general_regression_analysis_format_robust" / "general_regression_analysis.json").read_text()
            )
            self.assertEqual(report["run_intervals"]["base"]["estimate"], 1.0)
            self.assertEqual(
                report["generation_diagnostics"]["base"]["stored_answer_score_disagreements"],
                4,
            )
            expected = Path(str(artifact) + ".sha256").read_text().split()[0]
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), expected)
            with tarfile.open(artifact, "r:gz") as archive:
                self.assertIn(
                    "results/mmlu/general_regression_analysis_format_robust/general_regression_analysis.json",
                    archive.getnames(),
                )


if __name__ == "__main__":
    unittest.main()
