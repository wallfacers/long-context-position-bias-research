from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_transfer_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_transfer_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class TransferAnalysisTest(unittest.TestCase):
    def test_paired_bootstrap_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = []
            for run_index, run_name in enumerate(analysis.RUN_ORDER):
                path = root / f"{run_name}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for task_index, task in enumerate(analysis.TASKS):
                        for sample_index in range(4):
                            handle.write(
                                json.dumps(
                                    {
                                        "sample_id": f"{task}/{sample_index}",
                                        "task": task,
                                        "target_tokens": 3000 + task_index * 4000,
                                        "answer_score": min(1.0, 0.3 + 0.08 * run_index),
                                        "valid_json": True,
                                        "finish_reason": "stop",
                                        "output_tokens": 12,
                                    }
                                )
                                + "\n"
                            )
                arguments.extend(["--run", f"{run_name}={path}"])
            output = root / "analysis"
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    *arguments,
                    "--output-dir",
                    str(output),
                    "--bootstrap-replicates",
                    "100",
                    "--seed",
                    "11",
                ]
                self.assertEqual(analysis.main(), 0)
            finally:
                sys.argv = old_argv
            report = json.loads((output / "transfer_analysis.json").read_text())
            self.assertEqual(report["rows_per_run"]["base"], 12)
            self.assertEqual(report["bootstrap"]["replicates"], 100)
            self.assertGreater(
                report["contrasts"]["overall"]["paired_evidence_minus_base"]["estimate"],
                0,
            )
            self.assertTrue((output / "transfer_summary.csv").is_file())
            self.assertTrue((output / "paired_bootstrap_indices.jsonl.gz").is_file())


if __name__ == "__main__":
    unittest.main()
