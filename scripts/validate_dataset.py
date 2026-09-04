#!/usr/bin/env python3
"""Validate raw position groups or prepared SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_bias_research.io import read_jsonl


def validate_raw(
    rows: list[dict[str, Any]],
    *,
    schema_version: str,
    length_tolerance: float,
    position_tolerance: float,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_ids: set[str] = set()
    conditions: Counter[str] = Counter()
    for row in rows:
        if row["sample_id"] in sample_ids:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        sample_ids.add(row["sample_id"])
        groups[row["group_id"]].append(row)
        if row.get("evaluation_mode"):
            if int(row["actual_tokens"]) <= 0:
                raise ValueError(f"Invalid diagnostic length for {row['sample_id']}")
        else:
            length_error = abs(row["actual_tokens"] - row["target_tokens"]) / row["target_tokens"]
            if length_error > length_tolerance:
                raise ValueError(
                    f"Length error {length_error:.2%} for {row['sample_id']} exceeds "
                    f"{length_tolerance:.2%}"
                )
        if abs(row["actual_position"] - row["target_position"]) > position_tolerance:
            raise ValueError(
                f"Position error for {row['sample_id']}: target={row['target_position']:.3f}, "
                f"actual={row['actual_position']:.3f}"
            )
        for evidence_id in row["target"]["evidence_ids"]:
            if f"[Document {evidence_id}]" not in row["prompt"]:
                raise ValueError(f"Missing evidence id {evidence_id} in {row['sample_id']}")
        for quote in row["target"]["evidence_quotes"]:
            if quote not in row["prompt"]:
                raise ValueError(f"Missing exact quote in {row['sample_id']}: {quote}")
        conditions[
            f"{row['split']}|{row['task']}|{row['filler_type']}|{row['target_tokens']}"
        ] += 1

    position_counts: Counter[str] = Counter()
    for group_id, group in groups.items():
        invariants = {
            (
                json.dumps(sample["target"], sort_keys=True),
                sample["metadata"]["filler_fingerprint"],
                sample["task"],
                sample["filler_type"],
                sample["target_tokens"],
            )
            for sample in group
        }
        if len(invariants) != 1:
            raise ValueError(f"Group invariants differ within {group_id}")
        labels = [sample["position_label"] for sample in group]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate position labels within {group_id}")
        position_counts.update(labels)

    return {
        "schema": schema_version,
        "rows": len(rows),
        "groups": len(groups),
        "conditions": dict(sorted(conditions.items())),
        "position_counts": dict(sorted(position_counts.items())),
    }


def validate_matched_raw(rows: list[dict[str, Any]]) -> dict[str, Any]:
    facts: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        metadata = row.get("metadata", {})
        if metadata.get("training_design") != "matched-position-v1":
            raise ValueError(f"Missing matched design metadata in {row['sample_id']}")
        facts[metadata["fact_id"]][int(metadata["replica_index"])].append(row)

    replica_counts: Counter[int] = Counter()
    for fact_id, replicas in facts.items():
        replica_counts[len(replicas)] += 1
        fact_fingerprints = {
            row["metadata"]["fact_fingerprint"]
            for group in replicas.values()
            for row in group
        }
        targets = {
            json.dumps(row["target"], ensure_ascii=False, sort_keys=True)
            for group in replicas.values()
            for row in group
        }
        if len(fact_fingerprints) != 1 or len(targets) != 1:
            raise ValueError(f"Fact content changes across replicas for {fact_id}")
        position_grids = {
            tuple(sorted(row["position_label"] for row in group))
            for group in replicas.values()
        }
        if len(position_grids) != 1:
            raise ValueError(f"Position grids differ across replicas for {fact_id}")
        filler_fingerprints = {
            group[0]["metadata"]["filler_fingerprint"] for group in replicas.values()
        }
        if len(filler_fingerprints) != len(replicas):
            raise ValueError(f"Filler replicas are not unique for {fact_id}")

    return {
        "training_design": "matched-position-v1",
        "facts": len(facts),
        "replicas_per_fact": {
            str(count): facts_with_count
            for count, facts_with_count in sorted(replica_counts.items())
        },
    }


def validate_sft(
    rows: list[dict[str, Any]], *, schema_version: str
) -> dict[str, Any]:
    ids: set[str] = set()
    variants: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    total_tokens = 0
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate SFT id: {row['id']}")
        ids.add(row["id"])
        messages = row["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Invalid message roles in {row['id']}")
        try:
            response = json.loads(messages[-1]["content"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Assistant response is not JSON in {row['id']}") from exc
        if "answer" not in response:
            raise ValueError(f"Assistant response lacks answer in {row['id']}")
        expected_fields = {"answer", "evidence_ids", "evidence_quotes", "confidence"}
        if set(response) != expected_fields:
            raise ValueError(
                f"Assistant response fields differ in {row['id']}: {sorted(response)}"
            )
        metadata = row["metadata"]
        if schema_version == "position-sft-v1":
            if metadata["include_evidence_supervision"]:
                if not response.get("evidence_ids") or not response.get("evidence_quotes"):
                    raise ValueError(f"Evidence-supervised row lacks evidence in {row['id']}")
            elif response.get("evidence_ids") != [] or response.get("evidence_quotes") != []:
                raise ValueError(f"Answer-only row contains evidence content in {row['id']}")
        else:
            if metadata.get("training_design") != "matched-position-v1":
                raise ValueError(f"Missing matched design metadata in {row['id']}")
            supervision = metadata.get("supervision_mode")
            has_ids = bool(response.get("evidence_ids"))
            has_quotes = bool(response.get("evidence_quotes"))
            expected = {
                "answer": (False, False),
                "evidence_id": (True, False),
                "evidence": (True, True),
            }.get(supervision)
            if expected is None:
                raise ValueError(f"Unknown supervision mode in {row['id']}: {supervision}")
            if (has_ids, has_quotes) != expected:
                raise ValueError(
                    f"Supervision payload mismatch in {row['id']}: "
                    f"got {(has_ids, has_quotes)}, expected {expected}"
                )
        variants.update([metadata["variant"]])
        positions.update([metadata["position_label"]])
        total_tokens += int(metadata["actual_tokens"])
    return {
        "schema": schema_version,
        "rows": len(rows),
        "input_tokens": total_tokens,
        "variants": dict(sorted(variants.items())),
        "position_counts": dict(sorted(positions.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--length-tolerance", type=float, default=0.02)
    parser.add_argument("--position-tolerance", type=float, default=0.03)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reports: dict[str, Any] = {}
    for path in args.paths:
        rows = list(read_jsonl(path))
        if not rows:
            raise SystemExit(f"Empty dataset: {path}")
        schema = rows[0].get("schema_version")
        if any(row.get("schema_version") != schema for row in rows):
            raise SystemExit(f"Mixed schemas in {path}")
        if schema in {"position-group-v1", "position-group-v2"}:
            report = validate_raw(
                rows,
                schema_version=schema,
                length_tolerance=args.length_tolerance,
                position_tolerance=args.position_tolerance,
            )
            if schema == "position-group-v2":
                report.update(validate_matched_raw(rows))
        elif schema in {"position-sft-v1", "position-sft-v2"}:
            report = validate_sft(rows, schema_version=schema)
        else:
            raise SystemExit(f"Unsupported schema {schema!r} in {path}")
        reports[str(path)] = report

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for path, report in reports.items():
            print(f"OK {path}")
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
