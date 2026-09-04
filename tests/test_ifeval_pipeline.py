import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class IFEvalPipelineTest(unittest.TestCase):
    def test_frozen_dataset_matches_official_source(self):
        source = ROOT / "third_party/google-research/instruction_following_eval/data/input_data.jsonl"
        frozen = ROOT / "data/regression_ifeval/input_data.jsonl"
        manifest_path = ROOT / "data/regression_ifeval/manifest.json"
        if not source.is_file() or not frozen.is_file() or not manifest_path.is_file():
            self.skipTest(
                "IFEval payload is release-excluded; reconstruct pinned third-party data before this test"
            )
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        with source.open(encoding="utf-8") as handle:
            source_rows = [json.loads(line) for line in handle]
        with frozen.open(encoding="utf-8") as handle:
            frozen_rows = [json.loads(line) for line in handle]
        self.assertEqual(len(source_rows), 541)
        self.assertEqual(len(frozen_rows), 541)
        self.assertEqual(manifest["instruction_instances"], 834)
        self.assertEqual(
            [row["prompt"] for row in source_rows],
            [row["prompt"] for row in frozen_rows],
        )

    def test_metric_summary_uses_prompt_and_instruction_denominators(self):
        module = load_script("score_ifeval_results.py")
        template = {
            "a": {
                "strict_follow_all": True,
                "strict_follow_list": [True, True],
                "loose_follow_all": True,
                "loose_follow_list": [True, True],
            },
            "b": {
                "strict_follow_all": False,
                "strict_follow_list": [True, False, False],
                "loose_follow_all": False,
                "loose_follow_list": [True, True, False],
            },
        }
        scored = {run: template for run in module.RUN_ORDER}
        summary = module.summarize(scored, ["a", "b"])["base"]
        self.assertEqual(summary["strict_prompt"], 0.5)
        self.assertEqual(summary["strict_instruction"], 3 / 5)
        self.assertEqual(summary["loose_prompt"], 0.5)
        self.assertEqual(summary["loose_instruction"], 4 / 5)

    def test_generation_runner_keeps_raw_unconstrained_text(self):
        source = (ROOT / "scripts/evaluate_ifeval_suite_vllm.py").read_text(encoding="utf-8")
        self.assertIn("Duplicate sample ID", source)
        self.assertIn("Expected 541 IFEval prompts", source)
        self.assertNotIn("guided_decoding", source)
        self.assertNotIn("response_format", source)


if __name__ == "__main__":
    unittest.main()
