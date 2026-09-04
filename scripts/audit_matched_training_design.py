#!/usr/bin/env python3
"""Audit matched SFT variants for fact, exposure, filler, and position balance."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAIRING_MODES = ("independent", "paired")
SUPERVISION_MODES = ("answer", "evidence_id", "evidence")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "position-sft-v2":
                raise ValueError(f"Unexpected schema at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty variant: {path}")
    return rows


def variant_identity(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    identities = {
        (
            row["metadata"]["variant"],
            row["metadata"]["pairing_mode"],
            row["metadata"]["supervision_mode"],
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError(f"Mixed variant identities: {sorted(identities)}")
    return next(iter(identities))


def expected_response_shape(row: dict[str, Any], supervision: str) -> None:
    response = json.loads(row["messages"][-1]["content"])
    has_ids = bool(response.get("evidence_ids"))
    has_quotes = bool(response.get("evidence_quotes"))
    expected = {
        "answer": (False, False),
        "evidence_id": (True, False),
        "evidence": (True, True),
    }[supervision]
    if (has_ids, has_quotes) != expected:
        raise ValueError(
            f"Response supervision mismatch in {row['id']}: "
            f"got ids={has_ids}, quotes={has_quotes}, expected={expected}"
        )


def key_by_fact_replica(row: dict[str, Any]) -> tuple[str, int]:
    metadata = row["metadata"]
    return metadata["design_fact_id"], int(metadata["replica_index"])


def audit(
    paths: list[Path], *, max_token_budget_gap: float
) -> dict[str, Any]:
    variants: dict[tuple[str, str], list[dict[str, Any]]] = {}
    variant_names: set[str] = set()
    for path in paths:
        rows = read_rows(path)
        name, pairing, supervision = variant_identity(rows)
        if name != f"{pairing}_{supervision}":
            raise ValueError(f"Variant name does not match factors: {name}")
        if name in variant_names:
            raise ValueError(f"Duplicate variant input: {name}")
        variant_names.add(name)
        variants[(pairing, supervision)] = rows
        for row in rows:
            expected_response_shape(row, supervision)

    expected_keys = {
        (pairing, supervision)
        for pairing in PAIRING_MODES
        for supervision in SUPERVISION_MODES
    }
    if set(variants) != expected_keys:
        missing = sorted(expected_keys - set(variants))
        extra = sorted(set(variants) - expected_keys)
        raise ValueError(f"Variant matrix incomplete; missing={missing}, extra={extra}")

    report: dict[str, Any] = {
        "schema_version": "matched-sft-audit-v1",
        "status": "ok",
        "variants": {},
    }
    for pairing in PAIRING_MODES:
        reference = variants[(pairing, "answer")]
        reference_cells = {
            key_by_fact_replica(row): (
                row["metadata"]["position_label"],
                row["metadata"]["filler_fingerprint"],
                row["metadata"]["fact_fingerprint"],
                row["metadata"]["actual_tokens"],
            )
            for row in reference
        }
        if len(reference_cells) != len(reference):
            raise ValueError(f"Duplicate fact/replica cells in {pairing}")
        for supervision in SUPERVISION_MODES[1:]:
            candidate = variants[(pairing, supervision)]
            candidate_cells = {
                key_by_fact_replica(row): (
                    row["metadata"]["position_label"],
                    row["metadata"]["filler_fingerprint"],
                    row["metadata"]["fact_fingerprint"],
                    row["metadata"]["actual_tokens"],
                )
                for row in candidate
            }
            if candidate_cells != reference_cells:
                raise ValueError(
                    f"Supervision variants change inputs within pairing={pairing}"
                )

    independent = variants[("independent", "answer")]
    paired = variants[("paired", "answer")]
    independent_cells = {
        key_by_fact_replica(row): row["metadata"] for row in independent
    }
    paired_cells = {key_by_fact_replica(row): row["metadata"] for row in paired}
    if set(independent_cells) != set(paired_cells):
        raise ValueError("Pairing variants do not use the same fact/replica cells")
    for key in independent_cells:
        left = independent_cells[key]
        right = paired_cells[key]
        for field in ("fact_fingerprint", "filler_fingerprint", "actual_tokens"):
            if left[field] != right[field]:
                raise ValueError(f"Pairing variants differ in {field} for {key}")

    positions_by_pairing: dict[str, dict[str, set[str]]] = {}
    for pairing, rows in (("independent", independent), ("paired", paired)):
        positions: dict[str, set[str]] = defaultdict(set)
        exposures: Counter[str] = Counter()
        for row in rows:
            fact_id = row["metadata"]["design_fact_id"]
            positions[fact_id].add(row["metadata"]["position_label"])
            exposures[fact_id] += 1
        if len(set(exposures.values())) != 1:
            raise ValueError(f"Unequal exposures per fact in {pairing}")
        if pairing == "independent" and any(len(value) != 1 for value in positions.values()):
            raise ValueError("Independent facts must remain at one assigned position")
        if pairing == "paired":
            expected_position_count = next(iter(exposures.values()))
            if any(len(value) != expected_position_count for value in positions.values()):
                raise ValueError("Paired facts must cover every position exactly once")
        positions_by_pairing[pairing] = positions

    token_totals = {
        pairing: sum(int(row["metadata"]["actual_tokens"]) for row in rows)
        for pairing, rows in (("independent", independent), ("paired", paired))
    }
    mean_tokens = sum(token_totals.values()) / len(token_totals)
    token_gap = (max(token_totals.values()) - min(token_totals.values())) / max(
        mean_tokens, 1
    )
    if token_gap > max_token_budget_gap:
        raise ValueError(
            f"Pairing input-token gap {token_gap:.2%} exceeds {max_token_budget_gap:.2%}"
        )

    for (pairing, supervision), rows in sorted(variants.items()):
        exposures = Counter(row["metadata"]["design_fact_id"] for row in rows)
        report["variants"][f"{pairing}_{supervision}"] = {
            "rows": len(rows),
            "unique_facts": len(exposures),
            "exposures_per_fact": sorted(set(exposures.values())),
            "input_tokens": sum(int(row["metadata"]["actual_tokens"]) for row in rows),
            "positions": dict(
                sorted(Counter(row["metadata"]["position_label"] for row in rows).items())
            ),
            "unique_filler_views": len(
                {row["metadata"]["filler_fingerprint"] for row in rows}
            ),
        }
    report["pairing_comparison"] = {
        "same_fact_replica_cells": True,
        "same_fact_fingerprints": True,
        "same_filler_views": True,
        "input_token_totals": token_totals,
        "input_token_gap_fraction": token_gap,
        "independent_positions_per_fact": sorted(
            {len(value) for value in positions_by_pairing["independent"].values()}
        ),
        "paired_positions_per_fact": sorted(
            {len(value) for value in positions_by_pairing["paired"].values()}
        ),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-token-budget-gap", type=float, default=0.002)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit(args.paths, max_token_budget_gap=args.max_token_budget_gap)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
        print(f"Wrote {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
