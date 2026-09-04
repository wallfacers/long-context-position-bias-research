#!/usr/bin/env python3
"""Evaluate base plus multiple LoRA runs while loading the base model once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams

from evaluate_vllm import (
    RESPONSE_SCHEMA,
    batched,
    build_prompt,
    existing_ids,
    load_rows,
    score,
    sha256_directory,
    sha256_file,
    write_json_atomic,
)
from position_bias_research.chat_protocol import selected_protocol_for_tokenizer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_run(value: str) -> tuple[str, Path | None]:
    if "=" not in value:
        name = value.strip()
        adapter = None
    else:
        name, raw_path = value.split("=", 1)
        name = name.strip()
        adapter = Path(raw_path)
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
        raise argparse.ArgumentTypeError(f"Invalid run name: {name!r}")
    if adapter is not None and not adapter.is_dir():
        raise argparse.ArgumentTypeError(f"Missing adapter directory: {adapter}")
    return name, adapter


def selected_rows(rows: list[dict[str, Any]], maximum: int | None) -> list[dict[str, Any]]:
    if maximum is None or maximum >= len(rows):
        return rows
    if maximum <= 0:
        raise ValueError("--max-samples must be positive")

    # The generated data is ordered by position within each equivalence group.
    # A plain fixed-stride sample can alias that seven-row cycle and select only
    # one position.  Round-robin over experimental cells so a canary actually
    # exercises every position and task represented in the split.
    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("position_label"),
            row.get("task"),
            row.get("filler_type"),
            row.get("target_tokens"),
        )
        strata[key].append(row)
    ordered_keys = sorted(strata, key=lambda item: tuple(str(value) for value in item))
    chosen: list[dict[str, Any]] = []
    depth = 0
    while len(chosen) < maximum:
        added = False
        for key in ordered_keys:
            bucket = strata[key]
            if depth < len(bucket):
                chosen.append(bucket[depth])
                added = True
                if len(chosen) == maximum:
                    break
        if not added:
            break
        depth += 1
    return chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        metavar="NAME[=ADAPTER_DIR]",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.model.is_dir():
        raise SystemExit("--model must be a complete local directory")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    names = [name for name, _ in args.run]
    if len(names) != len(set(names)):
        raise SystemExit("Run names must be unique")
    rows = selected_rows(load_rows(args.data), args.max_samples)
    if not rows:
        raise SystemExit("No evaluation rows")
    data_hash = sha256_file(args.data)
    selected_ids = [row["sample_id"] for row in rows]
    selection_sha256 = hashlib.sha256(
        json.dumps(selected_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    model_manifest_path = args.model / "model_manifest.json"
    model_attestation_path = args.model / "model_integrity_attestation.json"
    if not model_manifest_path.is_file() or not model_attestation_path.is_file():
        raise SystemExit(
            "Evaluation requires model_manifest.json and "
            "model_integrity_attestation.json from the pinned staging gate"
        )
    model_manifest_sha256 = sha256_file(model_manifest_path)
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    model_attestation = json.loads(model_attestation_path.read_text(encoding="utf-8"))
    if (
        model_manifest.get("schema_version") != "local-model-manifest-v1"
        or model_attestation.get("schema_version")
        != "model-integrity-attestation-v1"
        or Path(model_attestation.get("model", "")).resolve() != args.model.resolve()
        or model_attestation.get("manifest_sha256") != model_manifest_sha256
        or model_attestation.get("revision") != model_manifest.get("revision")
    ):
        raise SystemExit("Pinned model manifest/attestation identity validation failed")
    for artifact in model_attestation.get("file_state", []):
        path = args.model / artifact["path"]
        if not path.is_file():
            raise SystemExit(f"Pinned model artifact is missing: {path}")
        stat = path.stat()
        if (
            stat.st_size != int(artifact["bytes"])
            or stat.st_mtime_ns != int(artifact["mtime_ns"])
        ):
            raise SystemExit(f"Pinned model artifact changed after attestation: {path}")
    tokenizer_payload = {
        "backend": tokenizer.backend_tokenizer.to_str(),
        "chat_template": tokenizer.chat_template,
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    tokenizer_fingerprint = hashlib.sha256(
        json.dumps(
            tokenizer_payload, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    chat_protocol = selected_protocol_for_tokenizer(tokenizer)
    chat_audit_path = args.model / "chat_protocol_audit.json"
    chat_protocol_audit_sha256 = (
        sha256_file(chat_audit_path) if chat_audit_path.is_file() else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_lengths: list[tuple[int, str]] = []
    for row in rows:
        rendered = build_prompt(tokenizer, row)
        token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        prompt_lengths.append((token_count, str(row["sample_id"])))
    prompt_lengths.sort()
    maximum_prompt_tokens, maximum_sample_id = prompt_lengths[-1]
    prompt_budget = args.max_model_len - args.max_new_tokens
    if maximum_prompt_tokens > prompt_budget:
        raise SystemExit(
            "Evaluation prompt exceeds the frozen context budget: "
            f"sample={maximum_sample_id} prompt_tokens={maximum_prompt_tokens} "
            f"max_allowed={prompt_budget} "
            f"(max_model_len={args.max_model_len}, max_new_tokens={args.max_new_tokens})"
        )
    length_audit = {
        "schema_version": "eval-prompt-length-audit-v1",
        "status": "validated",
        "data": str(args.data.resolve()),
        "data_sha256": data_hash,
        "model_revision": model_manifest.get("revision"),
        "model_manifest_sha256": model_manifest_sha256,
        "model_integrity_attestation_sha256": sha256_file(model_attestation_path),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "chat_protocol": chat_protocol,
        "chat_protocol_audit_sha256": chat_protocol_audit_sha256,
        "selected_samples": len(rows),
        "min_prompt_tokens": prompt_lengths[0][0],
        "median_prompt_tokens": prompt_lengths[len(prompt_lengths) // 2][0],
        "p95_prompt_tokens": prompt_lengths[
            int(0.95 * (len(prompt_lengths) - 1))
        ][0],
        "max_prompt_tokens": maximum_prompt_tokens,
        "max_prompt_sample_id": maximum_sample_id,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "unused_context_tokens_at_maximum": prompt_budget - maximum_prompt_tokens,
    }
    write_json_atomic(args.output_dir / "prompt-length-audit.json", length_audit)
    print(
        "prompt_length_audit=passed "
        f"max={maximum_prompt_tokens}+{args.max_new_tokens}<={args.max_model_len} "
        f"sample={maximum_sample_id}",
        flush=True,
    )
    adapters = [(name, path) for name, path in args.run if path is not None]
    llm_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tokenizer": str(args.model),
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    if args.max_num_seqs is not None:
        if args.max_num_seqs <= 0:
            raise SystemExit("--max-num-seqs must be positive")
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if adapters:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        structured_outputs=StructuredOutputsParams(json=RESPONSE_SCHEMA),
    )

    for lora_id, (run_name, adapter) in enumerate(args.run, start=1):
        output_path = args.output_dir / f"{run_name}.jsonl"
        metadata_path = output_path.with_suffix(".jsonl.run.json")
        identity = {
            "schema_version": "vllm-eval-run-v1",
            "run_name": run_name,
            "model": str(args.model.resolve()),
            "model_revision": model_manifest.get("revision"),
            "model_manifest_sha256": model_manifest_sha256,
            "model_integrity_attestation_sha256": sha256_file(
                model_attestation_path
            ),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "chat_protocol": chat_protocol,
            "chat_protocol_audit_sha256": chat_protocol_audit_sha256,
            "adapter": str(adapter.resolve()) if adapter else None,
            "adapter_sha256": sha256_directory(adapter) if adapter else None,
            "data": str(args.data.resolve()),
            "data_sha256": data_hash,
            "selected_samples": len(rows),
            "selection_sha256": selection_sha256,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_new_tokens": args.max_new_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_lora_rank": args.max_lora_rank,
            "enforce_eager": args.enforce_eager,
            "seed": args.seed,
        }
        if metadata_path.is_file():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if {key: previous.get(key) for key in identity} != identity:
                raise SystemExit(f"Run identity mismatch: {metadata_path}")
            started_at = previous["started_at"]
            previous_elapsed = float(previous.get("elapsed_seconds_total", 0.0))
            previous_invocations = int(previous.get("invocation_count", 0))
        else:
            started_at = utc_now()
            previous_elapsed = 0.0
            previous_invocations = 0
            write_json_atomic(metadata_path, identity | {"started_at": started_at})

        completed = existing_ids(output_path)
        pending = [row for row in rows if row["sample_id"] not in completed]
        print(
            f"run={run_name} selected={len(rows)} completed={len(rows) - len(pending)} "
            f"pending={len(pending)}",
            flush=True,
        )
        if not pending:
            continue
        lora_request = (
            LoRARequest(run_name, lora_id, str(adapter)) if adapter is not None else None
        )
        started = time.monotonic()
        processed = 0
        with output_path.open("a", encoding="utf-8") as handle:
            for batch in batched(pending, args.batch_size):
                prompts = [build_prompt(tokenizer, row) for row in batch]
                batch_started = time.monotonic()
                outputs = llm.generate(
                    prompts,
                    sampling,
                    lora_request=lora_request,
                    use_tqdm=False,
                )
                batch_seconds = time.monotonic() - batch_started
                for row, output in zip(batch, outputs, strict=True):
                    candidate = output.outputs[0]
                    generated = candidate.text
                    result = {
                        "schema_version": "position-eval-result-v1",
                        "run_name": run_name,
                        "evaluation_mode": row.get("evaluation_mode", "free"),
                        "sample_id": row["sample_id"],
                        "group_id": row["group_id"],
                        "task": row["task"],
                        "filler_type": row["filler_type"],
                        "target_tokens": row["target_tokens"],
                        "position_label": row["position_label"],
                        "target_position": row["target_position"],
                        "actual_position": row["actual_position"],
                        "target": row["target"],
                        "metadata": row.get("metadata", {}),
                        "generated_text": generated,
                        "finish_reason": candidate.finish_reason,
                        "output_tokens": len(candidate.token_ids),
                        "batch_wall_seconds": batch_seconds,
                    } | score(row, generated)
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                processed += len(batch)
                elapsed = time.monotonic() - started
                print(
                    f"run={run_name} saved={processed}/{len(pending)} "
                    f"sec/sample={elapsed / processed:.3f}",
                    flush=True,
                )
        elapsed = time.monotonic() - started
        write_json_atomic(
            metadata_path,
            identity
            | {
                "started_at": started_at,
                "last_finished_at": utc_now(),
                "samples_this_invocation": processed,
                "elapsed_seconds_this_invocation": elapsed,
                "seconds_per_sample_this_invocation": elapsed / processed,
                "elapsed_seconds_total": previous_elapsed + elapsed,
                "invocation_count": previous_invocations + 1,
                "status": "selection_complete",
            },
        )
    print(f"EVALUATION SUITE COMPLETE: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
