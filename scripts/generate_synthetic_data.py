#!/usr/bin/env python3
"""Generate deterministic position-equivalent KV and two-hop JSONL data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_bias_research.io import write_jsonl_atomic
from position_bias_research.synthetic_data import (
    SUPPORTED_FILLERS,
    SUPPORTED_TASKS,
    iter_synthetic_samples,
)
from position_bias_research.tokenization import load_token_counter


def csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def token_lengths(value: str) -> list[int]:
    parsed: list[int] = []
    for item in csv_values(value):
        normalized = item.lower().replace("_", "")
        multiplier = 1
        if normalized.endswith("k"):
            multiplier = 1024
            normalized = normalized[:-1]
        try:
            number = int(normalized) * multiplier
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid token length: {item}") from exc
        if number <= 0:
            raise argparse.ArgumentTypeError("token lengths must be positive")
        parsed.append(number)
    return parsed


def positions(value: str) -> list[float]:
    parsed: list[float] = []
    for item in csv_values(value):
        try:
            number = float(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid position: {item}") from exc
        if number > 1:
            number /= 100.0
        if number < 0 or number > 1:
            raise argparse.ArgumentTypeError("positions must be percentages or values in [0,1]")
        parsed.append(number)
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("positions must be unique")
    return parsed


def choices(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    parsed = csv_values(value)
    unknown = set(parsed) - set(allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {label}: {', '.join(sorted(unknown))}; allowed: {', '.join(allowed)}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--groups-per-condition", type=int, required=True)
    parser.add_argument(
        "--tasks",
        type=lambda value: choices(value, SUPPORTED_TASKS, "task"),
        default=list(SUPPORTED_TASKS),
    )
    parser.add_argument(
        "--fillers",
        type=lambda value: choices(value, SUPPORTED_FILLERS, "filler"),
        default=["neutral"],
    )
    parser.add_argument("--lengths", type=token_lengths, required=True)
    parser.add_argument("--positions", type=positions, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face tokenizer name, or 'whitespace' for smoke tests only.",
    )
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--words-per-document", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.groups_per_condition <= 0:
        raise SystemExit("--groups-per-condition must be positive")
    if args.words_per_document <= 0:
        raise SystemExit("--words-per-document must be positive")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.output}")

    counter = load_token_counter(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    rows = iter_synthetic_samples(
        split=args.split,
        groups_per_condition=args.groups_per_condition,
        tasks=args.tasks,
        filler_types=args.fillers,
        target_lengths=args.lengths,
        positions=args.positions,
        seed=args.seed,
        token_counter=counter,
        words_per_document=args.words_per_document,
    )
    count = write_jsonl_atomic(args.output, rows)
    condition_count = len(args.tasks) * len(args.fillers) * len(args.lengths)
    expected = args.groups_per_condition * condition_count * len(args.positions)
    if count != expected:
        raise RuntimeError(f"Generated {count} rows; expected {expected}")
    print(f"Wrote {count:,} rows to {args.output}")
    print(
        f"Groups: {args.groups_per_condition * condition_count:,}; "
        f"positions/group: {len(args.positions)}; tokenizer: {counter.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
