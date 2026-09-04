#!/usr/bin/env python3
"""Materialize strict block-complete SFT subsets from the formal data bank.

The fixed-100 exploratory runs sample only part of each 1,024-row dataset, so
they do not necessarily realize complete four-position blocks.  This script
selects whole fact blocks with an equal quota for every task × assigned-position
stratum.  With the frozen two tasks, four positions, three facts per stratum,
the result is 24 facts × four replicas = 96 rows.  A 96-step, batch-one run then
consumes the entire subset exactly once regardless of shuffled order.

Tokenized rows are selected from the already attested Arrow datasets.  The new
pretokenization manifests bind each 96-row source JSONL, every Arrow artifact,
the parent artifacts, and the selected indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_matched_training_design import audit as audit_matched_design
from position_bias_research.chat_protocol import NATIVE
from position_bias_research.io import write_jsonl_atomic


PAIRING_MODES = ("independent", "paired")
SUPERVISION_MODES = ("answer", "evidence_id", "evidence")
VARIANTS = tuple(
    f"{pairing}_{supervision}"
    for pairing in PAIRING_MODES
    for supervision in SUPERVISION_MODES
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
                raise ValueError(f"Unexpected SFT schema at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty SFT source: {path}")
    return rows


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds must contain unique integers")
    return seeds


def stable_order(seed: int, fact_id: str) -> str:
    return hashlib.sha256(f"block96:{seed}:{fact_id}".encode("utf-8")).hexdigest()


def fact_replica_key(row: dict[str, Any]) -> tuple[str, int]:
    metadata = row["metadata"]
    return metadata["design_fact_id"], int(metadata["replica_index"])


def selection_stratum(row: dict[str, Any]) -> tuple[str, str, int, str]:
    metadata = row["metadata"]
    assigned_position = metadata.get("assigned_position")
    if not assigned_position:
        raise ValueError("Independent reference row lacks assigned_position")
    return (
        metadata["task"],
        metadata["filler_type"],
        int(metadata["target_tokens"]),
        assigned_position,
    )


def select_block_indices(
    rows: list[dict[str, Any]], *, seed: int, facts_per_stratum: int
) -> tuple[list[int], dict[str, Any]]:
    if facts_per_stratum <= 0:
        raise ValueError("facts_per_stratum must be positive")
    facts: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        if metadata.get("pairing_mode") != "independent":
            raise ValueError("Block selection must use an independent reference")
        facts[metadata["design_fact_id"]].append((index, row))

    by_stratum: dict[tuple[str, str, int, str], list[str]] = defaultdict(list)
    replicas_per_fact: set[int] = set()
    for fact_id, group in facts.items():
        strata = {selection_stratum(row) for _, row in group}
        if len(strata) != 1:
            raise ValueError(f"Fact changes selection stratum: {fact_id}")
        replicas = {int(row["metadata"]["replica_index"]) for _, row in group}
        if len(replicas) != len(group):
            raise ValueError(f"Fact repeats a replica: {fact_id}")
        replicas_per_fact.add(len(group))
        by_stratum[next(iter(strata))].append(fact_id)
    if len(replicas_per_fact) != 1:
        raise ValueError("Facts have unequal replica counts")
    replicas = next(iter(replicas_per_fact))

    selected_facts: list[str] = []
    stratum_report: dict[str, Any] = {}
    for stratum in sorted(by_stratum):
        candidates = sorted(
            by_stratum[stratum], key=lambda fact_id: stable_order(seed, fact_id)
        )
        if len(candidates) < facts_per_stratum:
            raise ValueError(
                f"Stratum {stratum} has {len(candidates)} facts, needs {facts_per_stratum}"
            )
        chosen = candidates[:facts_per_stratum]
        selected_facts.extend(chosen)
        stratum_name = "|".join(map(str, stratum))
        stratum_report[stratum_name] = {
            "candidate_facts": len(candidates),
            "selected_facts": len(chosen),
            "selected_fact_ids_sha256": sha256_json(chosen),
        }

    selected_set = set(selected_facts)
    indices = [
        index
        for index, row in enumerate(rows)
        if row["metadata"]["design_fact_id"] in selected_set
    ]
    if len(indices) != len(selected_facts) * replicas:
        raise ValueError("Selected row count does not equal complete fact blocks")
    return indices, {
        "selection_algorithm": "sha256-stable-first-k-per-task-filler-length-assigned-position-v1",
        "facts_per_stratum": facts_per_stratum,
        "strata": stratum_report,
        "stratum_count": len(by_stratum),
        "selected_facts": len(selected_facts),
        "replicas_per_fact": replicas,
        "selected_rows": len(indices),
        "selected_indices": indices,
        "selected_indices_sha256": sha256_json(indices),
        "selected_fact_ids_sha256": sha256_json(sorted(selected_facts)),
    }


def artifact_manifest(output: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "pretokenization.json"
    ]


def file_record(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "lines": sum(1 for line in path.open("r", encoding="utf-8") if line.strip()),
        "sha256": sha256_file(path),
    }


def validate_parent_tokenization(
    source_path: Path, tokenized_path: Path, metadata: dict[str, Any]
) -> None:
    if metadata.get("schema_version") != "pretokenized-sft-v1":
        raise ValueError(f"Unexpected parent tokenization schema: {tokenized_path}")
    if metadata.get("source_sha256") != sha256_file(source_path):
        raise ValueError(f"Parent tokenization source differs: {tokenized_path}")
    for record in metadata.get("artifact_files", []):
        path = tokenized_path / record["path"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Parent tokenized artifact missing/changed: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Parent tokenized artifact hash differs: {path}")


def validate_existing_materialization(
    output_seed_root: Path,
    existing: dict[str, Any],
    *,
    seed: int,
    facts_per_stratum: int,
    expected_rows: int,
) -> None:
    """Fail closed before reusing an already materialized seed directory."""
    expected_positions = {
        "p000": expected_rows // 4,
        "p025": expected_rows // 4,
        "p050": expected_rows // 4,
        "p100": expected_rows // 4,
    }
    if not (
        existing.get("schema_version") == "block-complete-sft-seed-v1"
        and existing.get("status") == "validated"
        and int(existing.get("seed", -1)) == seed
        and int(existing.get("rows_per_variant", -1)) == expected_rows
        and int(existing.get("optimizer_steps", -1)) == expected_rows
        and int(existing.get("facts_per_stratum", -1)) == facts_per_stratum
        and existing.get("positions_per_variant") == expected_positions
        and existing.get("strict_realized_matching") is True
        and existing.get("complete_fact_blocks") is True
    ):
        raise ValueError(
            f"Existing output is not the expected validated subset: {output_seed_root}"
        )
    lineage = {
        "manifest_sha256": output_seed_root / "manifest.json",
        "matched_audit_sha256": output_seed_root / "matched-audit.json",
        "selection_sha256": output_seed_root / "selection.json",
    }
    for field, path in lineage.items():
        if (
            not path.is_file()
            or existing.get(field) != sha256_file(path)
        ):
            raise ValueError(
                f"Existing materialized lineage hash mismatch: seed={seed} field={field}"
            )


def token_lengths(dataset: Any) -> tuple[list[int], list[int]]:
    total: list[int] = []
    completion: list[int] = []
    for item in dataset:
        ids = item["input_ids"]
        mask = item["completion_mask"]
        if len(ids) != len(mask):
            raise ValueError("Tokenized IDs and completion mask lengths differ")
        suffix = sum(int(value) for value in mask)
        if list(mask) != [0] * (len(mask) - suffix) + [1] * suffix:
            raise ValueError("Completion mask is not a suffix")
        total.append(len(ids))
        completion.append(suffix)
    return total, completion


def materialize_seed(
    *,
    source_seed_root: Path,
    output_seed_root: Path,
    seed: int,
    facts_per_stratum: int,
    expected_rows: int,
    max_token_budget_gap: float,
) -> dict[str, Any]:
    try:
        from datasets import load_from_disk
    except ImportError as error:  # pragma: no cover - exercised in data-prep environments
        raise RuntimeError(
            "datasets is required to materialize tokenized block-complete subsets"
        ) from error
    completion_path = output_seed_root / "completion.json"
    if completion_path.is_file():
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        validate_existing_materialization(
            output_seed_root,
            existing,
            seed=seed,
            facts_per_stratum=facts_per_stratum,
            expected_rows=expected_rows,
        )
        return existing
    if output_seed_root.exists():
        raise ValueError(f"Refusing partial/existing output: {output_seed_root}")

    source_manifest_path = source_seed_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    reference_path = source_seed_root / "sft" / "independent_answer.jsonl"
    reference_rows = read_jsonl(reference_path)
    indices, selection = select_block_indices(
        reference_rows, seed=seed, facts_per_stratum=facts_per_stratum
    )
    if len(indices) != expected_rows:
        raise ValueError(
            f"Selection produced {len(indices)} rows, expected {expected_rows}"
        )

    temporary_root = output_seed_root.with_name(
        output_seed_root.name + f".tmp-{os.getpid()}"
    )
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir(parents=True)
    raw_outputs: list[Path] = []
    parent_records: dict[str, Any] = {}
    identity_reference = [fact_replica_key(reference_rows[index]) for index in indices]
    try:
        for variant in VARIANTS:
            source_path = source_seed_root / "sft" / f"{variant}.jsonl"
            parent_tokenized = source_seed_root / "tokenized" / variant
            parent_metadata_path = parent_tokenized / "pretokenization.json"
            rows = read_jsonl(source_path)
            if len(rows) != len(reference_rows):
                raise ValueError(f"Variant row count differs: {variant}")
            selected_rows = [rows[index] for index in indices]
            identities = [fact_replica_key(row) for row in selected_rows]
            if identities != identity_reference:
                raise ValueError(f"Variant row order/identity differs: {variant}")
            raw_output = temporary_root / "sft" / f"{variant}.jsonl"
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl_atomic(raw_output, selected_rows)
            raw_outputs.append(raw_output)

            parent_metadata = json.loads(
                parent_metadata_path.read_text(encoding="utf-8")
            )
            validate_parent_tokenization(source_path, parent_tokenized, parent_metadata)
            parent_dataset = load_from_disk(str(parent_tokenized))
            if len(parent_dataset) != len(rows):
                raise ValueError(f"Parent Arrow row count differs: {variant}")
            selected_dataset = parent_dataset.select(indices)
            tokenized_output = temporary_root / "tokenized" / variant
            selected_dataset.save_to_disk(str(tokenized_output))
            totals, completions = token_lengths(selected_dataset)
            metadata = {
                "schema_version": "pretokenized-sft-v1",
                "source": f"sft/{variant}.jsonl",
                "source_sha256": sha256_file(raw_output),
                "tokenizer": parent_metadata.get("tokenizer"),
                "tokenizer_revision": parent_metadata.get("tokenizer_revision"),
                "tokenizer_fingerprint": parent_metadata.get("tokenizer_fingerprint"),
                "chat_protocol": parent_metadata.get("chat_protocol") or NATIVE,
                "rows": len(selected_dataset),
                "max_length_limit": parent_metadata.get("max_length_limit"),
                "min_tokens": min(totals),
                "max_tokens": max(totals),
                "total_tokens": sum(totals),
                "min_completion_tokens": min(completions),
                "max_completion_tokens": max(completions),
                "total_completion_tokens": sum(completions),
                "lineage": {
                    "schema_version": "selected-parent-tokenization-lineage-v1",
                    "parent_source": parent_metadata.get("source"),
                    "parent_source_sha256": parent_metadata.get("source_sha256"),
                    "parent_pretokenization_manifest_sha256": sha256_file(
                        parent_metadata_path
                    ),
                    "selected_indices_sha256": selection["selected_indices_sha256"],
                    "selection_preserves_parent_row_order": True,
                },
                "artifact_files": artifact_manifest(tokenized_output),
            }
            write_json_atomic(tokenized_output / "pretokenization.json", metadata)
            parent_records[variant] = {
                "source_sha256": sha256_file(source_path),
                "pretokenization_manifest_sha256": sha256_file(parent_metadata_path),
            }

        audit_report = audit_matched_design(
            raw_outputs, max_token_budget_gap=max_token_budget_gap
        )
        if audit_report.get("status") != "ok":
            raise ValueError("Materialized matched-design audit failed")
        expected_positions = {
            "p000": expected_rows // 4,
            "p025": expected_rows // 4,
            "p050": expected_rows // 4,
            "p100": expected_rows // 4,
        }
        for variant, record in audit_report["variants"].items():
            if record["positions"] != expected_positions:
                raise ValueError(f"Position balance failed for {variant}")
            if record["exposures_per_fact"] != [4]:
                raise ValueError(f"Fact blocks are incomplete for {variant}")

        write_json_atomic(temporary_root / "selection.json", selection)
        write_json_atomic(temporary_root / "matched-audit.json", audit_report)
        files = [file_record(path, temporary_root) for path in raw_outputs]
        manifest = {
            "schema_version": "data-manifest-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "root": ".",
            "model": source_manifest.get("model"),
            "files": files,
            "totals": {
                "files": len(files),
                "lines": sum(record["lines"] for record in files),
                "bytes": sum(record["bytes"] for record in files),
            },
            "lineage": {
                "source_seed_root": source_seed_root.as_posix(),
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "parent_variants": parent_records,
            },
        }
        write_json_atomic(temporary_root / "manifest.json", manifest)
        completion = {
            "schema_version": "block-complete-sft-seed-v1",
            "status": "validated",
            "seed": seed,
            "rows_per_variant": expected_rows,
            "optimizer_steps": expected_rows,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "epochs_consumed": 1.0,
            "selected_facts": selection["selected_facts"],
            "replicas_per_fact": selection["replicas_per_fact"],
            "facts_per_stratum": facts_per_stratum,
            "positions_per_variant": expected_positions,
            "strict_realized_matching": True,
            "complete_fact_blocks": True,
            "selection_sha256": sha256_file(temporary_root / "selection.json"),
            "matched_audit_sha256": sha256_file(temporary_root / "matched-audit.json"),
            "manifest_sha256": sha256_file(temporary_root / "manifest.json"),
        }
        write_json_atomic(temporary_root / "completion.json", completion)
        output_seed_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root.replace(output_seed_root)
        return completion
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("20260825,20260826,20260827"),
    )
    parser.add_argument("--facts-per-stratum", type=int, default=3)
    parser.add_argument("--expected-rows", type=int, default=96)
    parser.add_argument("--max-token-budget-gap", type=float, default=0.002)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.source_root.is_dir():
        raise SystemExit(f"Missing source root: {args.source_root}")
    if args.expected_rows % 4:
        raise SystemExit("--expected-rows must be divisible by four positions")
    completions: dict[str, Any] = {}
    for seed in args.seeds:
        completions[str(seed)] = materialize_seed(
            source_seed_root=args.source_root / f"seed_{seed}",
            output_seed_root=args.output_root / f"seed_{seed}",
            seed=seed,
            facts_per_stratum=args.facts_per_stratum,
            expected_rows=args.expected_rows,
            max_token_budget_gap=args.max_token_budget_gap,
        )
        print(f"Validated strict block-complete subset for seed {seed}")
    report = {
        "schema_version": "block-complete-sft-multiseed-v1",
        "status": "validated",
        "source_root": args.source_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "seeds": args.seeds,
        "rows_per_variant": args.expected_rows,
        "facts_per_stratum": args.facts_per_stratum,
        "conditions_per_seed": len(VARIANTS),
        "strict_realized_matching": True,
        "seed_completions": completions,
    }
    write_json_atomic(args.output_root / "completion.json", report)
    print(f"Wrote strict block-complete data root: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
