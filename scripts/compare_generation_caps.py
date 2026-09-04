#!/usr/bin/env python3
"""Compare a balanced prefix evaluated with two generation-token caps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_hash(rows: list[dict[str, Any]]) -> str:
    rendered = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def metrics(rows: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "samples": len(rows),
        "finish_reason_length_count": sum(row["finish_reason"] == "length" for row in rows),
        "finish_reason_length_rate": sum(row["finish_reason"] == "length" for row in rows)
        / len(rows),
        "valid_json_count": sum(bool(row["valid_json"]) for row in rows),
        "valid_json_rate": sum(bool(row["valid_json"]) for row in rows) / len(rows),
        "answer_correct_count": sum(bool(row["answer_correct"]) for row in rows),
        "answer_accuracy": sum(bool(row["answer_correct"]) for row in rows) / len(rows),
        "selected_rows_sha256": canonical_hash(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-dir", type=Path, required=True)
    parser.add_argument("--higher-dir", type=Path, required=True)
    parser.add_argument("--run", default="base")
    parser.add_argument("--sample-limit", type=int, default=84)
    parser.add_argument("--expected-cells", type=int, default=84)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_limit <= 0:
        raise SystemExit("--sample-limit must be positive")

    lower_all = read_jsonl(args.lower_dir / f"{args.run}.jsonl")
    higher_all = read_jsonl(args.higher_dir / f"{args.run}.jsonl")
    if len(lower_all) < args.sample_limit or len(higher_all) < args.sample_limit:
        raise SystemExit("Both runs must contain the requested balanced prefix")
    lower = lower_all[: args.sample_limit]
    higher = higher_all[: args.sample_limit]
    lower_ids = [row["sample_id"] for row in lower]
    higher_ids = [row["sample_id"] for row in higher]
    if lower_ids != higher_ids or len(set(lower_ids)) != len(lower_ids):
        raise SystemExit("Generation-cap runs do not contain the same ordered sample IDs")
    cells = Counter(
        (
            row["task"],
            row["filler_type"],
            int(row["target_tokens"]),
            row["position_label"],
        )
        for row in higher
    )
    if len(cells) != args.expected_cells or set(cells.values()) != {1}:
        raise SystemExit(
            f"Expected one sample in each of {args.expected_cells} cells; got "
            f"cells={len(cells)} counts={dict(Counter(cells.values()))}"
        )
    lower_metadata = json.loads(
        (args.lower_dir / f"{args.run}.jsonl.run.json").read_text(encoding="utf-8")
    )
    higher_metadata = json.loads(
        (args.higher_dir / f"{args.run}.jsonl.run.json").read_text(encoding="utf-8")
    )
    lower_cap = int(lower_metadata["max_new_tokens"])
    higher_cap = int(higher_metadata["max_new_tokens"])
    if higher_cap <= lower_cap:
        raise SystemExit("Higher generation cap must be greater than lower cap")

    transitions = Counter(
        (lower_row["finish_reason"], higher_row["finish_reason"])
        for lower_row, higher_row in zip(lower, higher, strict=True)
    )
    resolved = [
        higher_row["sample_id"]
        for lower_row, higher_row in zip(lower, higher, strict=True)
        if lower_row["finish_reason"] == "length"
        and higher_row["finish_reason"] != "length"
    ]
    remaining = [
        higher_row["sample_id"]
        for higher_row in higher
        if higher_row["finish_reason"] == "length"
    ]
    payload = {
        "schema_version": "generation-cap-diagnostic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "balanced prefix protocol diagnostic; excluded from final treatment estimates",
        "run_name": args.run,
        "sample_ids_sha256": hashlib.sha256(
            json.dumps(lower_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "matrix": {
            "samples": len(higher),
            "cells": len(cells),
            "samples_per_cell": 1,
        },
        "lower_cap": metrics(lower, lower_cap),
        "higher_cap": metrics(higher, higher_cap),
        "finish_reason_transitions": {
            f"{start}_to_{end}": count
            for (start, end), count in sorted(transitions.items())
        },
        "lower_length_resolved_sample_ids": resolved,
        "higher_length_remaining_sample_ids": remaining,
        "decision": (
            "Use the higher cap only if it fits the model context window; inspect remaining "
            "length-finished outputs to distinguish pathological verbosity from valid JSON "
            "that still needs more output headroom."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
