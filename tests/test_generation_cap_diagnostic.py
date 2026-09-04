from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_generation_caps.py"


class GenerationCapDiagnosticTests(unittest.TestCase):
    def test_balanced_cap_comparison_records_resolved_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lower = root / "lower"
            higher = root / "higher"
            lower.mkdir()
            higher.mkdir()
            rows = []
            for index, position in enumerate(("p000", "p050")):
                rows.append(
                    {
                        "sample_id": f"s{index}",
                        "task": "kv",
                        "filler_type": "neutral",
                        "target_tokens": 8192,
                        "position_label": position,
                        "finish_reason": "length" if index == 0 else "stop",
                        "valid_json": index != 0,
                        "answer_correct": index != 0,
                    }
                )
            higher_rows = [dict(row) for row in rows]
            higher_rows[0].update(
                {"finish_reason": "stop", "valid_json": True, "answer_correct": True}
            )
            for directory, values, cap in ((lower, rows, 128), (higher, higher_rows, 176)):
                (directory / "base.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in values), encoding="utf-8"
                )
                (directory / "base.jsonl.run.json").write_text(
                    json.dumps({"max_new_tokens": cap}), encoding="utf-8"
                )
            output = root / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--lower-dir",
                    str(lower),
                    "--higher-dir",
                    str(higher),
                    "--sample-limit",
                    "2",
                    "--expected-cells",
                    "2",
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["lower_cap"]["finish_reason_length_count"], 1)
            self.assertEqual(report["higher_cap"]["finish_reason_length_count"], 0)
            self.assertEqual(report["lower_length_resolved_sample_ids"], ["s0"])


if __name__ == "__main__":
    unittest.main()
