from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "estimate_autodl_budget.py"
SPEC = importlib.util.spec_from_file_location("estimate_autodl_budget", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
budget = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = budget
SPEC.loader.exec_module(budget)


class BudgetEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = budget.load_config(ROOT / "configs" / "autodl_pilot_budget.json")

    def test_default_pilot_rounds_to_260_cny(self) -> None:
        estimate = budget.estimate_budget(self.config)
        self.assertAlmostEqual(
            estimate.total_gpu_hours,
            sum(item.gpu_hours for item in estimate.workloads),
        )
        self.assertAlmostEqual(
            estimate.single_queue_wall_hours,
            sum(item.wall_hours for item in estimate.workloads),
        )
        self.assertAlmostEqual(estimate.gpu_cost, 212.99377777777776)
        self.assertAlmostEqual(estimate.expected_total, 222.99377777777776)
        self.assertEqual(estimate.recommended_top_up, 260.0)

    def test_calibration_group_updates_all_training_workloads(self) -> None:
        updated = budget.apply_overrides(self.config, {}, {"qlora_train": 20.0})
        training = [
            item
            for item in updated["workloads"]
            if item["calibration_group"] == "qlora_train"
        ]
        self.assertTrue(training)
        self.assertTrue(all(item["seconds_per_unit"] == 20.0 for item in training))

    def test_skip_removes_completed_canary(self) -> None:
        full = budget.estimate_budget(self.config)
        remaining = budget.estimate_budget(
            self.config, skipped_workloads=["preflight_and_100_step_canary"]
        )
        self.assertLess(remaining.expected_total, full.expected_total)
        self.assertEqual(len(remaining.workloads), len(full.workloads) - 1)

    def test_remaining_units_collapses_in_progress_workload(self) -> None:
        updated = budget.apply_remaining_units(
            self.config, {"evaluation_remaining_20900_requests": 1234.0}
        )
        workload = next(
            item for item in updated["workloads"]
            if item["name"] == "evaluation_remaining_20900_requests"
        )
        self.assertEqual(workload["runs"], 1)
        self.assertEqual(workload["parallel_instances"], 1)
        self.assertEqual(workload["units_per_run"], 1234)
        self.assertEqual(workload["fixed_hours_per_run"], 0.0)

    def test_remaining_units_rejects_fractional_or_unknown_values(self) -> None:
        with self.assertRaises(ValueError):
            budget.apply_remaining_units(
                self.config, {"evaluation_remaining_20900_requests": 1.5}
            )
        with self.assertRaises(ValueError):
            budget.apply_remaining_units(self.config, {"missing": 1.0})

    def test_reads_measured_canary_seconds(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "CANARY_COMPLETE.json"
            path.write_text(
                json.dumps({"seconds_per_step_this_invocation": 17.25}),
                encoding="utf-8",
            )
            self.assertEqual(
                budget.measured_seconds(path, "seconds_per_step_this_invocation"),
                17.25,
            )

    def test_timezone_aware_start_time_drives_single_queue_eta(self) -> None:
        estimate = budget.estimate_budget(self.config)
        start = budget.parse_start_time("2026-08-28T23:42:27+08:00")
        expected, upper = budget.completion_times(estimate, start)
        self.assertGreater(expected, start)
        self.assertGreater(upper, expected)
        payload = json.loads(budget.as_json(self.config, estimate, start))
        self.assertEqual(payload["schedule_start"], start.isoformat())
        self.assertEqual(payload["expected_single_queue_finish"], expected.isoformat())
        self.assertEqual(
            payload["contingency_single_queue_finish"], upper.isoformat()
        )
        self.assertAlmostEqual(
            payload["total_gpu_hours"],
            sum(item.gpu_hours for item in estimate.workloads),
        )

    def test_start_time_requires_timezone(self) -> None:
        with self.assertRaises(ValueError):
            budget.parse_start_time("2026-08-28T23:42:27")
        self.assertEqual(
            budget.parse_start_time("2026-08-28T15:42:27Z"),
            datetime.fromisoformat("2026-08-28T15:42:27+00:00"),
        )

    def test_formal_matched_budget_rounds_to_350_cny(self) -> None:
        config = budget.load_config(
            ROOT / "configs" / "autodl_formal_matched_budget.json"
        )
        estimate = budget.estimate_budget(config)
        self.assertGreater(estimate.expected_total, 250.0)
        self.assertLess(estimate.expected_total, 300.0)
        self.assertEqual(estimate.recommended_top_up, 350.0)

    def test_top_tier_remaining_budget_stays_within_user_envelope(self) -> None:
        config = budget.load_config(
            ROOT / "configs" / "autodl_top_tier_completion_budget.json"
        )
        estimate = budget.estimate_budget(config)
        self.assertGreaterEqual(estimate.recommended_top_up, 300.0)
        self.assertLessEqual(estimate.recommended_top_up, 500.0)

    def test_strict_block96_budget_stays_within_user_envelope(self) -> None:
        config = budget.load_config(
            ROOT / "configs" / "autodl_strict_block96_budget.json"
        )
        estimate = budget.estimate_budget(config)
        self.assertGreater(estimate.expected_total, 350.0)
        self.assertLess(estimate.expected_total, 425.0)
        self.assertLessEqual(estimate.budget_ceiling, 500.0)
        self.assertEqual(estimate.recommended_top_up, 500.0)


if __name__ == "__main__":
    unittest.main()
