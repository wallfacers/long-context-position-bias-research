#!/usr/bin/env python3
"""Build matched pairing × supervision SFT variants from a v2 training bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_bias_research.io import read_jsonl, write_jsonl_atomic


PAIRING_MODES = ("independent", "paired")
SUPERVISION_MODES = ("answer", "evidence_id", "evidence")
VARIANTS = tuple(
    f"{pairing}_{supervision}"
    for pairing in PAIRING_MODES
    for supervision in SUPERVISION_MODES
)


def stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def condition_key(sample: dict[str, Any]) -> tuple[str, str, int]:
    return sample["task"], sample["filler_type"], int(sample["target_tokens"])


def assistant_payload(sample: dict[str, Any], supervision: str) -> str:
    target = sample["target"]
    if supervision == "answer":
        evidence_ids: list[str] = []
        evidence_quotes: list[str] = []
    elif supervision == "evidence_id":
        evidence_ids = list(target["evidence_ids"])
        evidence_quotes = []
    elif supervision == "evidence":
        evidence_ids = list(target["evidence_ids"])
        evidence_quotes = list(target["evidence_quotes"])
    else:
        raise ValueError(f"Unsupported supervision mode: {supervision}")
    return json.dumps(
        {
            "answer": target["answer"],
            "evidence_ids": evidence_ids,
            "evidence_quotes": evidence_quotes,
            "confidence": 1.0,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def to_sft(
    sample: dict[str, Any],
    *,
    variant: str,
    pairing: str,
    supervision: str,
    design_fact_id: str,
    assigned_position: str | None,
) -> dict[str, Any]:
    raw_metadata = sample["metadata"]
    return {
        "schema_version": "position-sft-v2",
        "id": f"{variant}:{sample['sample_id']}",
        "messages": [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["prompt"]},
            {
                "role": "assistant",
                "content": assistant_payload(sample, supervision),
            },
        ],
        "metadata": {
            "training_design": "matched-position-v1",
            "variant": variant,
            "pairing_mode": pairing,
            "supervision_mode": supervision,
            "raw_sample_id": sample["sample_id"],
            "group_id": sample["group_id"],
            "fact_id": raw_metadata["fact_id"],
            "design_fact_id": design_fact_id,
            "fact_fingerprint": raw_metadata["fact_fingerprint"],
            "filler_fingerprint": raw_metadata["filler_fingerprint"],
            "replica_index": int(raw_metadata["replica_index"]),
            "task": sample["task"],
            "filler_type": sample["filler_type"],
            "target_tokens": sample["target_tokens"],
            "actual_tokens": sample["actual_tokens"],
            "target_position": sample["target_position"],
            "actual_position": sample["actual_position"],
            "position_label": sample["position_label"],
            "assigned_position": assigned_position,
            "tokenizer": sample["tokenizer"],
            "position_grouped": pairing == "paired",
            "include_evidence_supervision": supervision != "answer",
            "include_quote_supervision": supervision == "evidence",
        },
    }


def _index_bank(
    rows: Iterable[dict[str, Any]],
) -> dict[
    tuple[str, str, int],
    dict[str, dict[int, dict[str, dict[str, Any]]]],
]:
    bank: dict[
        tuple[str, str, int],
        dict[str, dict[int, dict[str, dict[str, Any]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    seen_ids: set[str] = set()
    for row in rows:
        if row.get("schema_version") != "position-group-v2":
            raise ValueError(f"Unexpected raw schema: {row.get('schema_version')}")
        if row["sample_id"] in seen_ids:
            raise ValueError(f"Duplicate raw sample: {row['sample_id']}")
        seen_ids.add(row["sample_id"])
        metadata = row.get("metadata", {})
        if metadata.get("training_design") != "matched-position-v1":
            raise ValueError(f"Unexpected training design in {row['sample_id']}")
        key = condition_key(row)
        fact_id = metadata["fact_id"]
        replica = int(metadata["replica_index"])
        label = row["position_label"]
        if label in bank[key][fact_id][replica]:
            raise ValueError(f"Duplicate bank cell: {key}/{fact_id}/{replica}/{label}")
        bank[key][fact_id][replica][label] = row
    if not bank:
        raise ValueError("Input contains no matched-bank rows")
    return bank


def build_variants(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    max_token_budget_gap: float,
) -> dict[str, list[dict[str, Any]]]:
    bank = _index_bank(rows)
    selected: dict[str, list[tuple[dict[str, Any], str, str | None]]] = {
        pairing: [] for pairing in PAIRING_MODES
    }

    expected_positions: tuple[str, ...] | None = None
    expected_replicas: tuple[int, ...] | None = None
    for condition in sorted(bank):
        facts = bank[condition]
        ordered_facts = sorted(
            facts,
            key=lambda fact_id: stable_order(seed, f"{condition}:{fact_id}"),
        )
        for fact_rank, fact_id in enumerate(ordered_facts):
            replicas = tuple(sorted(facts[fact_id]))
            position_sets = {
                tuple(sorted(facts[fact_id][replica])) for replica in replicas
            }
            if len(position_sets) != 1:
                raise ValueError(f"Position grids differ across replicas for {fact_id}")
            positions = next(iter(position_sets))
            if len(replicas) != len(positions):
                raise ValueError(
                    f"Matched design needs equal replicas and positions for {fact_id}: "
                    f"{len(replicas)} vs {len(positions)}"
                )
            if expected_positions is None:
                expected_positions = positions
                expected_replicas = replicas
            elif positions != expected_positions or replicas != expected_replicas:
                raise ValueError("All conditions must use the same replicas and position grid")

            design_fact_id = (
                f"{condition[0]}|{condition[1]}|{condition[2]}|{fact_id}"
            )
            assigned_position = positions[fact_rank % len(positions)]
            for replica_rank, replica in enumerate(replicas):
                paired_position = positions[replica_rank]
                paired_row = facts[fact_id][replica][paired_position]
                independent_row = facts[fact_id][replica][assigned_position]
                selected["paired"].append((paired_row, design_fact_id, None))
                selected["independent"].append(
                    (independent_row, design_fact_id, assigned_position)
                )

    raw_token_totals = {
        pairing: sum(int(row["actual_tokens"]) for row, _, _ in samples)
        for pairing, samples in selected.items()
    }
    mean = sum(raw_token_totals.values()) / len(raw_token_totals)
    gap = (max(raw_token_totals.values()) - min(raw_token_totals.values())) / max(mean, 1)
    if gap > max_token_budget_gap:
        raise ValueError(
            f"Input token budget gap {gap:.2%} exceeds {max_token_budget_gap:.2%}: "
            f"{raw_token_totals}"
        )

    variants: dict[str, list[dict[str, Any]]] = {}
    for pairing in PAIRING_MODES:
        for supervision in SUPERVISION_MODES:
            variant = f"{pairing}_{supervision}"
            variants[variant] = [
                to_sft(
                    row,
                    variant=variant,
                    pairing=pairing,
                    supervision=supervision,
                    design_fact_id=design_fact_id,
                    assigned_position=assigned_position,
                )
                for row, design_fact_id, assigned_position in selected[pairing]
            ]
    return variants


def summarize(variants: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "matched-sft-design-report-v1",
        "training_design": "matched-position-v1",
        "variants": {},
    }
    for name, rows in sorted(variants.items()):
        exposures = Counter(row["metadata"]["design_fact_id"] for row in rows)
        report["variants"][name] = {
            "rows": len(rows),
            "unique_facts": len(exposures),
            "min_exposures_per_fact": min(exposures.values()),
            "max_exposures_per_fact": max(exposures.values()),
            "input_tokens": sum(int(row["metadata"]["actual_tokens"]) for row in rows),
            "positions": dict(
                sorted(Counter(row["metadata"]["position_label"] for row in rows).items())
            ),
            "tasks": dict(sorted(Counter(row["metadata"]["task"] for row in rows).items())),
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-token-budget-gap", type=float, default=0.002)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = {name: args.output_dir / f"{name}.jsonl" for name in VARIANTS}
    report_path = args.output_dir / "matched-design.json"
    existing = [path for path in [*outputs.values(), report_path] if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite existing files: " + ", ".join(map(str, existing))
        )

    variants = build_variants(
        read_jsonl(args.input),
        seed=args.seed,
        max_token_budget_gap=args.max_token_budget_gap,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in variants.items():
        count = write_jsonl_atomic(outputs[name], rows)
        tokens = sum(int(row["metadata"]["actual_tokens"]) for row in rows)
        print(f"{name:28} rows={count:,} input_tokens={tokens:,} -> {outputs[name]}")
    report = summarize(variants)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(f"Wrote matched design report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
