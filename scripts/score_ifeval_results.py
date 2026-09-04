#!/usr/bin/env python3
"""Score IFEval generations with official verifiers and paired bootstrap inference."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


VARIANTS = (
    "independent_answer",
    "independent_evidence_id",
    "independent_evidence",
    "paired_answer",
    "paired_evidence_id",
    "paired_evidence",
)
RUN_ORDER = ("base", *VARIANTS)
CONTRASTS = {
    **{
        f"{variant}_minus_base": {variant: 1.0, "base": -1.0}
        for variant in VARIANTS
    },
    **{
        f"paired_minus_independent_at_{supervision}": {
            f"paired_{supervision}": 1.0,
            f"independent_{supervision}": -1.0,
        }
        for supervision in ("answer", "evidence_id", "evidence")
    },
}
METRICS = ("strict_prompt", "strict_instruction", "loose_prompt", "loose_instruction")


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be CANONICAL_NAME=JSONL")
    name, path = value.split("=", 1)
    if name not in RUN_ORDER:
        raise argparse.ArgumentTypeError(f"Unknown run: {name}")
    return name, Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * probability
    lower, upper = math.floor(coordinate), math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_p_two_sided(values: Sequence[float]) -> float:
    denominator = len(values) + 1
    return min(
        1.0,
        2
        * min(
            (sum(value <= 0 for value in values) + 1) / denominator,
            (sum(value >= 0 for value in values) + 1) / denominator,
        ),
    )


def holm_adjust(items: Sequence[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        adjusted[name] = running
    return adjusted


def weighted(values: dict[str, float], weights: dict[str, float]) -> float:
    return sum(values[run] * coefficient for run, coefficient in weights.items())


def summarize(scored: dict[str, dict[str, dict[str, Any]]], selected: list[str]) -> dict[str, dict[str, float]]:
    output = {}
    for run in RUN_ORDER:
        rows = [scored[run][sample_id] for sample_id in selected]
        output[run] = {
            "strict_prompt": sum(row["strict_follow_all"] for row in rows) / len(rows),
            "strict_instruction": sum(sum(row["strict_follow_list"]) for row in rows)
            / sum(len(row["strict_follow_list"]) for row in rows),
            "loose_prompt": sum(row["loose_follow_all"] for row in rows) / len(rows),
            "loose_instruction": sum(sum(row["loose_follow_list"]) for row in rows)
            / sum(len(row["loose_follow_list"]) for row in rows),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--noninferiority-margin", type=float, default=0.02)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("Use at least 100 bootstrap replicates")
    paths = dict(args.run)
    if set(paths) != set(RUN_ORDER) or len(args.run) != len(RUN_ORDER):
        raise SystemExit("Exactly one generation file is required for each canonical run")
    sys.path.insert(0, str(args.official_root.parent.resolve()))
    try:
        import langdetect

        # langdetect is otherwise nondeterministic for short outputs. This does
        # not change the official verifier rules; it freezes their randomness.
        langdetect.DetectorFactory.seed = 0
        from instruction_following_eval import evaluation_lib
    except ImportError as exc:
        raise SystemExit(
            "Official IFEval dependencies/code are unavailable; install its pinned requirements"
        ) from exc
    official_inputs = evaluation_lib.read_prompt_list(str(args.input_data))
    if len(official_inputs) != 541:
        raise SystemExit(f"Expected 541 official IFEval inputs, found {len(official_inputs)}")
    prompt_to_input = {item.prompt: item for item in official_inputs}
    raw_by_run = {run: read_jsonl(paths[run]) for run in RUN_ORDER}
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    reference_ids: set[str] | None = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run, rows in raw_by_run.items():
        if len(rows) != 541:
            raise SystemExit(f"{run}: expected 541 generations, found {len(rows)}")
        table = {str(row["sample_id"]): row for row in rows}
        if len(table) != len(rows):
            raise SystemExit(f"{run}: duplicate sample IDs")
        if reference_ids is None:
            reference_ids = set(table)
        elif set(table) != reference_ids:
            raise SystemExit(f"{run}: sample IDs differ from base")
        scored[run] = {}
        scored_path = args.output_dir / f"{run}.scored.jsonl"
        with scored_path.open("w", encoding="utf-8") as handle:
            for sample_id in sorted(table):
                row = table[sample_id]
                official_input = prompt_to_input.get(row["prompt"])
                if official_input is None:
                    raise SystemExit(f"{run}/{sample_id}: prompt differs from official input")
                responses = {row["prompt"]: row["response"]}
                strict = evaluation_lib.test_instruction_following_strict(
                    official_input, responses
                )
                loose = evaluation_lib.test_instruction_following_loose(
                    official_input, responses
                )
                record = {
                    "sample_id": sample_id,
                    "key": row["key"],
                    "instruction_id_list": official_input.instruction_id_list,
                    "strict_follow_all": strict.follow_all_instructions,
                    "strict_follow_list": strict.follow_instruction_list,
                    "loose_follow_all": loose.follow_all_instructions,
                    "loose_follow_list": loose.follow_instruction_list,
                }
                scored[run][sample_id] = record
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    assert reference_ids is not None
    ids = sorted(reference_ids)
    by_instruction_count: dict[int, list[str]] = defaultdict(list)
    for sample_id in ids:
        by_instruction_count[
            len(scored["base"][sample_id]["instruction_id_list"])
        ].append(sample_id)
    point = summarize(scored, ids)
    run_samples = {(run, metric): [] for run in RUN_ORDER for metric in METRICS}
    contrast_samples = {
        (contrast, metric): [] for contrast in CONTRASTS for metric in METRICS
    }
    rng = random.Random(args.seed)
    indices_path = args.output_dir / "paired_bootstrap_indices.jsonl.gz"
    with gzip.open(indices_path, "wt", encoding="utf-8") as handle:
        for replicate in range(args.bootstrap_replicates):
            selected = []
            serialized = {}
            for count, candidates in sorted(by_instruction_count.items()):
                indices = [rng.randrange(len(candidates)) for _ in candidates]
                selected.extend(candidates[index] for index in indices)
                serialized[str(count)] = indices
            handle.write(
                json.dumps(
                    {"replicate": replicate, "instruction_count_indices": serialized},
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            values = summarize(scored, selected)
            for run in RUN_ORDER:
                for metric in METRICS:
                    run_samples[(run, metric)].append(values[run][metric])
            for contrast, weights in CONTRASTS.items():
                for metric in METRICS:
                    contrast_samples[(contrast, metric)].append(
                        weighted({run: values[run][metric] for run in RUN_ORDER}, weights)
                    )
    intervals = {
        run: {
            metric: {
                "estimate": point[run][metric],
                "ci95_low": percentile(run_samples[(run, metric)], 0.025),
                "ci95_high": percentile(run_samples[(run, metric)], 0.975),
            }
            for metric in METRICS
        }
        for run in RUN_ORDER
    }
    contrasts = {}
    for metric in METRICS:
        raw_p = []
        for name, weights in CONTRASTS.items():
            samples = contrast_samples[(name, metric)]
            p_value = bootstrap_p_two_sided(samples)
            raw_p.append((name, p_value))
            contrasts.setdefault(name, {})[metric] = {
                "estimate": weighted(
                    {run: point[run][metric] for run in RUN_ORDER}, weights
                ),
                "ci95_low": percentile(samples, 0.025),
                "ci95_high": percentile(samples, 0.975),
                "bootstrap_p_two_sided": p_value,
            }
        for name, value in holm_adjust(raw_p).items():
            contrasts[name][metric]["holm_adjusted_p"] = value
    margin = args.noninferiority_margin
    noninferiority = {
        run: {
            "margin": -margin,
            "difference": contrasts[f"{run}_minus_base"]["strict_prompt"]["estimate"],
            "ci95_low": contrasts[f"{run}_minus_base"]["strict_prompt"]["ci95_low"],
            "passes_if_ci95_low_above_margin": (
                contrasts[f"{run}_minus_base"]["strict_prompt"]["ci95_low"] > -margin
            ),
        }
        for run in VARIANTS
    }
    report = {
        "schema_version": "ifeval-regression-analysis-v1",
        "official_revision": "041338718b4e8151372fd63677104c65b73a0a4e",
        "prompts": len(ids),
        "instruction_instances": sum(
            len(scored["base"][sample_id]["instruction_id_list"]) for sample_id in ids
        ),
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "unit": "prompt paired across runs and stratified by number of constraints",
            "indices": indices_path.name,
        },
        "run_intervals": intervals,
        "contrasts": contrasts,
        "noninferiority_to_base_strict_prompt": noninferiority,
        "generation_diagnostics": {
            run: {
                "finish_reason_counts": dict(
                    sorted(Counter(str(row.get("finish_reason", "unknown")) for row in rows).items())
                ),
                "mean_output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows)
                / len(rows),
            }
            for run, rows in raw_by_run.items()
        },
    }
    (args.output_dir / "ifeval_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote official IFEval scoring and paired analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
