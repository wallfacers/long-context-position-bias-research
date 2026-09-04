#!/usr/bin/env python3
"""Build equal-budget independent/paired x answer/evidence SFT variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_bias_research.io import read_jsonl, write_jsonl_atomic
VARIANTS = (
    "independent_answer",
    "paired_answer",
    "independent_evidence",
    "paired_evidence",
)


def condition_key(sample: dict[str, Any]) -> tuple[str, str, int]:
    return sample["task"], sample["filler_type"], int(sample["target_tokens"])


def stable_order(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def allocate(total: int, keys: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], int]:
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    base, remainder = divmod(total, len(keys))
    return {key: base + (index < remainder) for index, key in enumerate(keys)}


def assistant_payload(sample: dict[str, Any], include_evidence: bool) -> str:
    target = sample["target"]
    if include_evidence:
        payload = {
            "answer": target["answer"],
            "evidence_ids": target["evidence_ids"],
            "evidence_quotes": target["evidence_quotes"],
            "confidence": 1.0,
        }
    else:
        # Keep the input instruction identical across supervision ablations while
        # withholding evidence content. Empty arrays satisfy the response schema
        # without teaching evidence IDs or quotes.
        payload = {
            "answer": target["answer"],
            "evidence_ids": [],
            "evidence_quotes": [],
            "confidence": 1.0,
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def to_sft(sample: dict[str, Any], variant: str) -> dict[str, Any]:
    include_evidence = variant.endswith("evidence")
    return {
        "schema_version": "position-sft-v1",
        "id": f"{variant}:{sample['sample_id']}",
        "messages": [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["prompt"]},
            {
                "role": "assistant",
                "content": assistant_payload(sample, include_evidence),
            },
        ],
        "metadata": {
            "variant": variant,
            "raw_sample_id": sample["sample_id"],
            "group_id": sample["group_id"],
            "task": sample["task"],
            "filler_type": sample["filler_type"],
            "target_tokens": sample["target_tokens"],
            "actual_tokens": sample["actual_tokens"],
            "target_position": sample["target_position"],
            "actual_position": sample["actual_position"],
            "position_label": sample["position_label"],
            "tokenizer": sample["tokenizer"],
            "position_grouped": variant.startswith("paired"),
            "include_evidence_supervision": include_evidence,
        },
    }


def flatten(groups: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [sample for group in groups for sample in group]


def build_variants(
    rows: Iterable[dict[str, Any]],
    *,
    paired_groups: int,
    independent_groups: int,
    seed: int,
    max_token_budget_gap: float,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("schema_version") != "position-group-v1":
            raise ValueError(f"Unexpected raw schema: {row.get('schema_version')}")
        groups[row["group_id"]].append(row)
    if not groups:
        raise ValueError("Input contains no groups")

    buckets: dict[tuple[str, str, int], list[list[dict[str, Any]]]] = defaultdict(list)
    for group in groups.values():
        group.sort(key=lambda sample: float(sample["target_position"]))
        positions = [sample["target_position"] for sample in group]
        if len(positions) != len(set(positions)):
            raise ValueError(f"Duplicate position in group {group[0]['group_id']}")
        buckets[condition_key(group[0])].append(group)

    keys = sorted(buckets)
    paired_allocation = allocate(paired_groups, keys)
    independent_allocation = allocate(independent_groups, keys)
    paired_selected: list[list[dict[str, Any]]] = []
    independent_selected: list[dict[str, Any]] = []

    for key in keys:
        bucket = sorted(
            buckets[key], key=lambda group: stable_order(seed, group[0]["group_id"])
        )
        paired_count = paired_allocation[key]
        independent_count = independent_allocation[key]
        required = paired_count + independent_count
        if len(bucket) < required:
            raise ValueError(
                f"Condition {key} has {len(bucket)} groups but needs {required}"
            )
        paired_bucket = bucket[:paired_count]
        independent_bucket = bucket[paired_count:required]
        paired_selected.extend(paired_bucket)
        for index, group in enumerate(independent_bucket):
            independent_selected.append(group[index % len(group)])

    paired_samples = flatten(paired_selected)
    if len(paired_samples) != len(independent_selected):
        raise ValueError(
            "Equal sequence budgets require independent_groups == "
            "paired_groups * positions_per_group; got "
            f"{len(independent_selected)} independent and {len(paired_samples)} paired samples"
        )

    raw_by_variant = {
        "independent_answer": independent_selected,
        "paired_answer": paired_samples,
        "independent_evidence": independent_selected,
        "paired_evidence": paired_samples,
    }
    token_totals = {
        name: sum(int(sample["actual_tokens"]) for sample in samples)
        for name, samples in raw_by_variant.items()
    }
    minimum = min(token_totals.values())
    maximum = max(token_totals.values())
    mean = sum(token_totals.values()) / len(token_totals)
    gap = (maximum - minimum) / max(mean, 1)
    if gap > max_token_budget_gap:
        raise ValueError(
            f"Token budget gap {gap:.2%} exceeds limit {max_token_budget_gap:.2%}: "
            f"{token_totals}"
        )

    return {
        name: [to_sft(sample, name) for sample in samples]
        for name, samples in raw_by_variant.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paired-groups", type=int, default=250)
    parser.add_argument("--independent-groups", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-token-budget-gap", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = {
        variant: args.output_dir / f"{variant}.jsonl" for variant in VARIANTS
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite existing files: " + ", ".join(map(str, existing))
        )
    variants = build_variants(
        read_jsonl(args.input),
        paired_groups=args.paired_groups,
        independent_groups=args.independent_groups,
        seed=args.seed,
        max_token_budget_gap=args.max_token_budget_gap,
    )
    for name, rows in variants.items():
        count = write_jsonl_atomic(outputs[name], rows)
        tokens = sum(int(row["metadata"]["actual_tokens"]) for row in rows)
        print(f"{name:22} rows={count:,} input_tokens={tokens:,} -> {outputs[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
