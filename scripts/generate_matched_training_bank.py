#!/usr/bin/env python3
"""Generate a matched fact/replica/position bank for formal SFT ablations."""

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
    iter_matched_training_bank,
)
from position_bias_research.tokenization import load_token_counter


def csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def choices(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    values = csv_values(value)
    unknown = set(values) - set(allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {label}: {', '.join(sorted(unknown))}; allowed: {', '.join(allowed)}"
        )
    return values


def token_lengths(value: str) -> list[int]:
    parsed: list[int] = []
    for item in csv_values(value):
        normalized = item.lower().replace("_", "")
        multiplier = 1024 if normalized.endswith("k") else 1
        if multiplier != 1:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--facts-per-condition", type=int, default=128)
    parser.add_argument("--replicas-per-fact", type=int, default=4)
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
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--words-per-document", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.facts_per_condition <= 0:
        raise SystemExit("--facts-per-condition must be positive")
    if args.replicas_per_fact <= 0:
        raise SystemExit("--replicas-per-fact must be positive")
    if args.replicas_per_fact != len(args.positions):
        raise SystemExit(
            "The matched design requires --replicas-per-fact to equal the number of positions"
        )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.output}")

    counter = load_token_counter(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    rows = iter_matched_training_bank(
        split=args.split,
        facts_per_condition=args.facts_per_condition,
        replicas_per_fact=args.replicas_per_fact,
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
    expected = (
        args.facts_per_condition
        * condition_count
        * args.replicas_per_fact
        * len(args.positions)
    )
    if count != expected:
        raise RuntimeError(f"Generated {count} rows; expected {expected}")
    print(f"Wrote {count:,} matched-bank rows to {args.output}")
    print(
        f"facts={args.facts_per_condition * condition_count:,} "
        f"replicas/fact={args.replicas_per_fact} positions/replica={len(args.positions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
