#!/usr/bin/env python3
"""Reconstruct and audit the examples consumed by fixed-step SFT runs.

The formal datasets contain more rows than the preregistered 100 optimizer
steps consume.  Transformers 5.x creates a RandomSampler and Accelerate 1.x
replaces it with SeedableRandomSampler.  With one process, batch size one, and
gradient accumulation one, the first ``steps`` indices are therefore the first
``steps`` values of a seed-initialized ``torch.randperm``.  This audit checks
that statement against Accelerate's sampler implementation, verifies the
raw-to-Arrow lineage, and measures the *realized* rather than nominal pairing
intervention.

Run this script in the exact training environment so that its PyTorch and
Accelerate versions match the recorded runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PAIRING_MODES = ("independent", "paired")
SUPERVISION_MODES = ("answer", "evidence_id", "evidence")
VARIANTS = tuple(
    f"{pairing}_{supervision}"
    for pairing in PAIRING_MODES
    for supervision in SUPERVISION_MODES
)
NATIVE_CHAT_PROTOCOL = "native-system-user-assistant"
SUPPORTED_CHAT_PROTOCOLS = {
    NATIVE_CHAT_PROTOCOL,
    "merge-system-into-first-user-v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
        raise ValueError(f"Empty SFT data: {path}")
    return rows


def counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))
    }


def fact_replica_key(row: dict[str, Any]) -> tuple[str, int]:
    metadata = row["metadata"]
    return metadata["design_fact_id"], int(metadata["replica_index"])


def prompt_messages_hash(row: dict[str, Any]) -> str:
    return sha256_json(row["messages"][:-1])


def normalized_chat_protocol(metadata: dict[str, Any]) -> str:
    protocol = metadata.get("chat_protocol") or NATIVE_CHAT_PROTOCOL
    if protocol not in SUPPORTED_CHAT_PROTOCOLS:
        raise ValueError(f"Unsupported pretokenization chat protocol: {protocol}")
    return str(protocol)


def validate_pretokenization_lineage(
    *, source_path: Path, tokenized_path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    if metadata.get("schema_version") != "pretokenized-sft-v1":
        raise ValueError(f"Unexpected pretokenization schema: {tokenized_path}")
    source_hash = sha256_file(source_path)
    if metadata.get("source_sha256") != source_hash:
        raise ValueError(f"Pretokenization source hash differs: {source_path}")
    for record in metadata.get("artifact_files", []):
        artifact = tokenized_path / record["path"]
        if not artifact.is_file():
            raise ValueError(f"Missing tokenized artifact: {artifact}")
        if artifact.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Tokenized artifact size differs: {artifact}")
        if sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"Tokenized artifact hash differs: {artifact}")
    return {
        "source": metadata.get("source"),
        "source_sha256": source_hash,
        "pretokenization_manifest_sha256": sha256_file(
            tokenized_path / "pretokenization.json"
        ),
        "tokenizer": metadata.get("tokenizer"),
        "tokenizer_revision": metadata.get("tokenizer_revision"),
        "tokenizer_fingerprint": metadata.get("tokenizer_fingerprint"),
        "chat_protocol": normalized_chat_protocol(metadata),
        "artifact_file_count": len(metadata.get("artifact_files", [])),
    }


def reconstruct_sampler_indices(row_count: int, seed: int, steps: int) -> tuple[list[int], dict[str, str]]:
    if not 0 < steps <= row_count:
        raise ValueError(f"steps must be in [1, {row_count}], got {steps}")
    try:
        import accelerate
        import torch
        from accelerate.data_loader import SeedableRandomSampler
    except ImportError as error:  # pragma: no cover - exercised in the GPU environment
        raise RuntimeError(
            "Run the sampler audit in the training environment with torch and accelerate"
        ) from error

    generator = torch.Generator(device="cpu")
    direct = torch.randperm(row_count, generator=generator.manual_seed(seed)).tolist()
    sampler = SeedableRandomSampler(
        range(row_count),
        replacement=False,
        num_samples=row_count,
        generator=torch.Generator(device="cpu"),
        data_seed=seed,
    )
    accelerated = list(sampler)
    if direct != accelerated:
        raise ValueError("torch.randperm and Accelerate sampler orders differ")
    return [int(index) for index in direct[:steps]], {
        "torch": str(torch.__version__),
        "accelerate": str(accelerate.__version__),
    }


def exposure_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fact[row["metadata"]["design_fact_id"]].append(row)

    exposure_histogram = Counter(len(group) for group in by_fact.values())
    facts_with_multiple_exposures = 0
    facts_with_cross_position_exposure = 0
    facts_with_all_four_positions = 0
    within_fact_pairs = 0
    cross_position_pairs = 0
    selected_examples_in_multi_exposure_facts = 0
    for group in by_fact.values():
        count = len(group)
        positions = [row["metadata"]["position_label"] for row in group]
        unique_positions = set(positions)
        if count >= 2:
            facts_with_multiple_exposures += 1
            selected_examples_in_multi_exposure_facts += count
        if len(unique_positions) >= 2:
            facts_with_cross_position_exposure += 1
        if len(unique_positions) == 4:
            facts_with_all_four_positions += 1
        for left in range(count):
            for right in range(left + 1, count):
                within_fact_pairs += 1
                if positions[left] != positions[right]:
                    cross_position_pairs += 1

    return {
        "unique_facts": len(by_fact),
        "exposures_per_fact_histogram": {
            str(key): int(value) for key, value in sorted(exposure_histogram.items())
        },
        "facts_with_multiple_exposures": facts_with_multiple_exposures,
        "facts_with_cross_position_exposure": facts_with_cross_position_exposure,
        "facts_with_all_four_positions": facts_with_all_four_positions,
        "selected_examples_in_multi_exposure_facts": selected_examples_in_multi_exposure_facts,
        "within_fact_pairs": within_fact_pairs,
        "cross_position_pairs": cross_position_pairs,
    }


def summarize_selected_rows(
    rows: list[dict[str, Any]], tokenized_rows: list[dict[str, Any]], indices: list[int]
) -> dict[str, Any]:
    if len(rows) != len(tokenized_rows):
        raise ValueError("Raw and tokenized row counts differ")
    selected_rows = [rows[index] for index in indices]
    selected_tokens = [tokenized_rows[index] for index in indices]
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    total_lengths: list[int] = []
    prompt_prefix_hashes: list[str] = []
    for row, tokenized in zip(selected_rows, selected_tokens, strict=True):
        input_ids = [int(value) for value in tokenized["input_ids"]]
        completion_mask = [int(value) for value in tokenized["completion_mask"]]
        if len(input_ids) != len(completion_mask):
            raise ValueError(f"Token/mask length differs for {row['id']}")
        completion_length = sum(completion_mask)
        prompt_length = len(input_ids) - completion_length
        if completion_mask != [0] * prompt_length + [1] * completion_length:
            raise ValueError(f"Completion mask is not a single suffix for {row['id']}")
        prompt_lengths.append(prompt_length)
        completion_lengths.append(completion_length)
        total_lengths.append(len(input_ids))
        prompt_prefix_hashes.append(sha256_json(input_ids[:prompt_length]))

    metadata = [row["metadata"] for row in selected_rows]
    identities = [list(fact_replica_key(row)) for row in selected_rows]
    summary = {
        "selected_rows": len(selected_rows),
        "selected_indices_sha256": sha256_json(indices),
        "ordered_fact_replica_identities_sha256": sha256_json(identities),
        "ordered_raw_sample_ids_sha256": sha256_json(
            [item["raw_sample_id"] for item in metadata]
        ),
        "ordered_prompt_messages_sha256": sha256_json(
            [prompt_messages_hash(row) for row in selected_rows]
        ),
        "ordered_tokenized_prompt_prefixes_sha256": sha256_json(prompt_prefix_hashes),
        "positions": counter_dict(item["position_label"] for item in metadata),
        "replicas": counter_dict(int(item["replica_index"]) for item in metadata),
        "tasks": counter_dict(item["task"] for item in metadata),
        "filler_types": counter_dict(item["filler_type"] for item in metadata),
        "target_tokens": counter_dict(int(item["target_tokens"]) for item in metadata),
        "raw_actual_tokens": sum(int(item["actual_tokens"]) for item in metadata),
        "tokenized_prompt_tokens": sum(prompt_lengths),
        "tokenized_completion_tokens": sum(completion_lengths),
        "tokenized_total_tokens": sum(total_lengths),
        "min_total_tokens": min(total_lengths),
        "max_total_tokens": max(total_lengths),
    }
    summary.update(exposure_statistics(selected_rows))
    return summary


def validate_supervision_siblings(
    rows_by_variant: dict[str, list[dict[str, Any]]], indices: list[int], pairing: str
) -> dict[str, Any]:
    siblings = [rows_by_variant[f"{pairing}_{mode}"] for mode in SUPERVISION_MODES]
    reference = siblings[0]
    for candidate in siblings[1:]:
        for index in indices:
            left, right = reference[index], candidate[index]
            if fact_replica_key(left) != fact_replica_key(right):
                raise ValueError(f"Supervision changes identity at index {index}")
            if left["messages"][:-1] != right["messages"][:-1]:
                raise ValueError(f"Supervision changes prompt at index {index}")
            for field in (
                "raw_sample_id",
                "position_label",
                "fact_fingerprint",
                "filler_fingerprint",
                "actual_tokens",
            ):
                if left["metadata"][field] != right["metadata"][field]:
                    raise ValueError(f"Supervision changes {field} at index {index}")
    return {
        "same_ordered_fact_replica_identities": True,
        "same_raw_samples": True,
        "same_prompts": True,
    }


def pairing_comparison(
    independent: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    indices: list[int],
    summaries: dict[str, dict[str, Any]],
    *,
    max_prompt_token_gap: float,
) -> dict[str, Any]:
    left = [independent[index] for index in indices]
    right = [paired[index] for index in indices]
    same_identities = [fact_replica_key(row) for row in left] == [
        fact_replica_key(row) for row in right
    ]
    same_fact_fingerprints = all(
        a["metadata"]["fact_fingerprint"] == b["metadata"]["fact_fingerprint"]
        for a, b in zip(left, right, strict=True)
    )
    same_filler_fingerprints = all(
        a["metadata"]["filler_fingerprint"] == b["metadata"]["filler_fingerprint"]
        for a, b in zip(left, right, strict=True)
    )
    same_nuisance_counts = all(
        summaries["independent_answer"][field]
        == summaries["paired_answer"][field]
        for field in ("tasks", "filler_types", "target_tokens", "replicas")
    )
    left_prompt = int(summaries["independent_answer"]["tokenized_prompt_tokens"])
    right_prompt = int(summaries["paired_answer"]["tokenized_prompt_tokens"])
    prompt_mean = max((left_prompt + right_prompt) / 2, 1)
    prompt_gap = abs(left_prompt - right_prompt) / prompt_mean
    positions_exact = (
        summaries["independent_answer"]["positions"]
        == summaries["paired_answer"]["positions"]
    )
    paired_blocks_complete = (
        summaries["paired_answer"]["facts_with_all_four_positions"]
        == summaries["paired_answer"]["unique_facts"]
    )
    strict = all(
        (
            same_identities,
            same_fact_fingerprints,
            same_filler_fingerprints,
            same_nuisance_counts,
            positions_exact,
            prompt_gap <= max_prompt_token_gap,
            paired_blocks_complete,
        )
    )
    return {
        "same_ordered_fact_replica_identities": same_identities,
        "same_fact_fingerprints": same_fact_fingerprints,
        "same_filler_fingerprints": same_filler_fingerprints,
        "same_task_filler_length_replica_counts": same_nuisance_counts,
        "position_histograms_exactly_equal": positions_exact,
        "prompt_token_totals": {
            "independent": left_prompt,
            "paired": right_prompt,
        },
        "prompt_token_gap_fraction": prompt_gap,
        "prompt_token_gap_within_threshold": prompt_gap <= max_prompt_token_gap,
        "paired_blocks_complete_for_every_selected_fact": paired_blocks_complete,
        "strict_realized_fixed_step_matching": strict,
    }


def validate_training_run(
    output_root: Path, seed: int, variant: str, expected_steps: int
) -> dict[str, Any]:
    root = output_root / f"seed_{seed}" / variant
    config_path = root / "run_config.json"
    completion_path = root / "CANARY_COMPLETE.json"
    if not config_path.is_file() or not completion_path.is_file():
        raise ValueError(f"Missing fixed-step run evidence: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    arguments = config.get("arguments", {})
    expected_suffix = f"seed_{seed}/tokenized/{variant}"
    if not str(arguments.get("data", "")).replace("\\", "/").endswith(expected_suffix):
        raise ValueError(f"Run data path differs for {root}")
    if int(arguments.get("seed", -1)) != seed:
        raise ValueError(f"Run seed differs for {root}")
    if int(arguments.get("stop_after_steps", -1)) != expected_steps:
        raise ValueError(f"Stopping step differs for {root}")
    if int(arguments.get("gradient_accumulation_steps", -1)) != 1:
        raise ValueError(f"Gradient accumulation is not one for {root}")
    if int(completion.get("global_step", -1)) != expected_steps:
        raise ValueError(f"Completion step differs for {root}")
    if int(config.get("checkpoint_step_before_run", -1)) != 0:
        raise ValueError(f"Run did not start from the beginning: {root}")
    return {
        "run_config_sha256": sha256_file(config_path),
        "completion_sha256": sha256_file(completion_path),
        "global_step": expected_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "world_size": 1,
        "samples_consumed": expected_steps,
        "started_from_step": 0,
    }


def audit_seed(
    *,
    data_root: Path,
    seed: int,
    steps: int,
    training_output_root: Path | None,
    max_prompt_token_gap: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    seed_root = data_root / f"seed_{seed}"
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    tokenized_by_variant: dict[str, list[dict[str, Any]]] = {}
    lineage: dict[str, Any] = {}
    row_count: int | None = None
    for variant in VARIANTS:
        source_path = seed_root / "sft" / f"{variant}.jsonl"
        tokenized_path = seed_root / "tokenized" / variant
        metadata_path = tokenized_path / "pretokenization.json"
        if not source_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"Missing raw/tokenized variant: {seed}/{variant}")
        rows = read_jsonl(source_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        lineage[variant] = validate_pretokenization_lineage(
            source_path=source_path,
            tokenized_path=tokenized_path,
            metadata=metadata,
        )
        try:
            from datasets import load_from_disk
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("datasets is required to inspect Arrow rows") from error
        dataset = load_from_disk(str(tokenized_path))
        if set(dataset.column_names) != {"input_ids", "completion_mask"}:
            raise ValueError(f"Unexpected tokenized columns: {tokenized_path}")
        tokenized_rows = [dataset[index] for index in range(len(dataset))]
        if int(metadata.get("rows", -1)) != len(rows) or len(dataset) != len(rows):
            raise ValueError(f"Raw/tokenized/manifest row counts differ: {variant}")
        if row_count is None:
            row_count = len(rows)
        elif row_count != len(rows):
            raise ValueError("Variant row counts differ")
        rows_by_variant[variant] = rows
        tokenized_by_variant[variant] = tokenized_rows

    assert row_count is not None
    indices, versions = reconstruct_sampler_indices(row_count, seed, steps)
    summaries = {
        variant: summarize_selected_rows(
            rows_by_variant[variant], tokenized_by_variant[variant], indices
        )
        for variant in VARIANTS
    }
    sibling_checks = {
        pairing: validate_supervision_siblings(rows_by_variant, indices, pairing)
        for pairing in PAIRING_MODES
    }
    comparison = pairing_comparison(
        rows_by_variant["independent_answer"],
        rows_by_variant["paired_answer"],
        indices,
        summaries,
        max_prompt_token_gap=max_prompt_token_gap,
    )
    training_runs = None
    if training_output_root is not None:
        training_runs = {
            variant: validate_training_run(
                training_output_root, seed, variant, expected_steps=steps
            )
            for variant in VARIANTS
        }
    report: dict[str, Any] = {
        "seed": seed,
        "dataset_rows_per_variant": row_count,
        "optimizer_steps": steps,
        "samples_consumed": steps,
        "dataset_fraction_consumed": steps / row_count,
        "sampler": {
            "implementation": "accelerate.data_loader.SeedableRandomSampler",
            "first_epoch_equivalence": "torch.randperm(N, generator.manual_seed(data_seed))",
            "data_seed": seed,
            "selected_indices": indices,
            "selected_indices_sha256": sha256_json(indices),
            "torch_accelerate_orders_equal": True,
        },
        "lineage": lineage,
        "supervision_sibling_checks": sibling_checks,
        "variants": summaries,
        "pairing_comparison": comparison,
    }
    if training_runs is not None:
        report["training_run_evidence"] = training_runs
    return report, versions


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds must contain unique integers")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("20260825,20260826,20260827"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--training-output-root", type=Path)
    parser.add_argument("--max-prompt-token-gap", type=float, default=0.002)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"Missing data root: {args.data_root}")
    seed_reports: dict[str, Any] = {}
    versions: dict[str, str] | None = None
    for seed in args.seeds:
        report, seed_versions = audit_seed(
            data_root=args.data_root,
            seed=seed,
            steps=args.steps,
            training_output_root=args.training_output_root,
            max_prompt_token_gap=args.max_prompt_token_gap,
        )
        if versions is None:
            versions = seed_versions
        elif versions != seed_versions:
            raise ValueError("Library versions changed during audit")
        seed_reports[str(seed)] = report

    strict_by_seed = {
        seed: bool(report["pairing_comparison"]["strict_realized_fixed_step_matching"])
        for seed, report in seed_reports.items()
    }
    all_identity_matched = all(
        report["pairing_comparison"]["same_ordered_fact_replica_identities"]
        for report in seed_reports.values()
    )
    all_strict = all(strict_by_seed.values())
    payload = {
        "schema_version": "realized-training-subset-audit-v1",
        "status": "validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "libraries": versions or {},
        "data_root": args.data_root.as_posix(),
        "training_output_root": (
            args.training_output_root.as_posix() if args.training_output_root else None
        ),
        "seeds": args.seeds,
        "steps": args.steps,
        "max_prompt_token_gap": args.max_prompt_token_gap,
        "seed_reports": seed_reports,
        "claim_assessment": {
            "actual_sampler_reconstructed": True,
            "raw_to_tokenized_lineage_validated": True,
            "same_fact_replica_identities_across_pairing_modes_all_seeds": all_identity_matched,
            "strict_realized_fixed_step_matching_by_seed": strict_by_seed,
            "strict_realized_fixed_step_matching_all_seeds": all_strict,
            "interpretation": (
                "The fixed-step subsets satisfy the strict realized matching claim."
                if all_strict
                else "The full datasets are matched, but the realized fixed-step subsets are not block-complete and/or exactly position balanced. Materialize a deterministic block-complete subset before any strict causal pairing claim."
            ),
            "recommended_action": (
                "retain_current_runs"
                if all_strict
                else "retrain_from_materialized_block_complete_subsets"
            ),
        },
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload["claim_assessment"], indent=2, sort_keys=True))
    print(f"Wrote realized training-subset audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
