#!/usr/bin/env python3
"""Evaluate base plus LoRA variants on IFEval with unconstrained text generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from position_bias_research.chat_protocol import apply_chat_template

from evaluate_vllm import batched, existing_ids, sha256_directory, sha256_file, write_json_atomic


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_run(value: str) -> tuple[str, Path | None]:
    if "=" in value:
        name, path = value.split("=", 1)
        adapter = Path(path)
    else:
        name, adapter = value, None
    if not name or (adapter is not None and not adapter.is_dir()):
        raise argparse.ArgumentTypeError(f"Invalid run: {value}")
    return name, adapter


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "ifeval-prompt-v1":
                raise ValueError(f"Unexpected schema at {path}:{line_number}")
            if row["sample_id"] in seen:
                raise ValueError(f"Duplicate sample ID: {row['sample_id']}")
            seen.add(row["sample_id"])
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if not args.model.is_dir() or args.batch_size <= 0 or args.max_num_seqs <= 0:
        raise SystemExit("Invalid model or batch configuration")
    names = [name for name, _ in args.run]
    if len(names) != len(set(names)):
        raise SystemExit("Run names must be unique")
    rows = read_rows(args.data)
    if len(rows) != 541:
        raise SystemExit(f"Expected 541 IFEval prompts, found {len(rows)}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    prompts = [
        apply_chat_template(
            tokenizer,
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    selection_sha256 = hashlib.sha256(
        json.dumps([row["sample_id"] for row in rows], separators=(",", ":")).encode()
    ).hexdigest()
    adapters = [(name, path) for name, path in args.run if path is not None]
    llm_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tokenizer": str(args.model),
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
        "enforce_eager": True,
    }
    if adapters:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_hash = sha256_file(args.data)
    for lora_id, (run_name, adapter) in enumerate(args.run, 1):
        output = args.output_dir / f"{run_name}.jsonl"
        metadata = output.with_suffix(".jsonl.run.json")
        identity = {
            "schema_version": "ifeval-vllm-run-v1",
            "run_name": run_name,
            "model": str(args.model.resolve()),
            "adapter": str(adapter.resolve()) if adapter else None,
            "adapter_sha256": sha256_directory(adapter) if adapter else None,
            "data_sha256": data_hash,
            "selection_sha256": selection_sha256,
            "samples": len(rows),
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        }
        if metadata.is_file():
            previous = json.loads(metadata.read_text(encoding="utf-8"))
            if {key: previous.get(key) for key in identity} != identity:
                raise SystemExit(f"Run identity mismatch: {metadata}")
            started_at = previous["started_at"]
            previous_elapsed = float(
                previous.get(
                    "elapsed_seconds_total",
                    previous.get("elapsed_seconds_this_invocation", 0.0),
                )
            )
        else:
            started_at = utc_now()
            previous_elapsed = 0.0
            previous = identity | {"started_at": started_at}
            write_json_atomic(metadata, previous)
        completed = existing_ids(output)
        pending_indices = [
            index for index, row in enumerate(rows) if row["sample_id"] not in completed
        ]
        print(
            f"run={run_name} selected={len(rows)} completed={len(rows)-len(pending_indices)} "
            f"pending={len(pending_indices)}",
            flush=True,
        )
        if not pending_indices:
            if previous.get("status") != "selection_complete":
                write_json_atomic(
                    metadata,
                    previous
                    | {
                        "selected_samples": len(rows),
                        "elapsed_seconds_total": previous_elapsed,
                        "status": "selection_complete",
                    },
                )
            continue
        lora_request = LoRARequest(run_name, lora_id, str(adapter)) if adapter else None
        started = time.monotonic()
        processed = 0
        with output.open("a", encoding="utf-8") as handle:
            for batch_indices in batched(pending_indices, args.batch_size):
                batch_prompts = [prompts[index] for index in batch_indices]
                request_started = time.monotonic()
                outputs = llm.generate(
                    batch_prompts, sampling, lora_request=lora_request, use_tqdm=False
                )
                wall_seconds = time.monotonic() - request_started
                for index, generated in zip(batch_indices, outputs, strict=True):
                    candidate = generated.outputs[0]
                    row = rows[index]
                    handle.write(
                        json.dumps(
                            {
                                "schema_version": "ifeval-generation-v1",
                                "run_name": run_name,
                                "sample_id": row["sample_id"],
                                "key": row["key"],
                                "prompt": row["prompt"],
                                "response": candidate.text,
                                "finish_reason": candidate.finish_reason,
                                "output_tokens": len(candidate.token_ids),
                                "batch_wall_seconds": wall_seconds,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
                processed += len(batch_indices)
                print(
                    f"run={run_name} saved={processed}/{len(pending_indices)} "
                    f"sec/sample={(time.monotonic()-started)/processed:.3f}",
                    flush=True,
                )
        elapsed = time.monotonic() - started
        write_json_atomic(
            metadata,
            identity
            | {
                "started_at": started_at,
                "last_finished_at": utc_now(),
                "samples_this_invocation": processed,
                "elapsed_seconds_this_invocation": elapsed,
                "selected_samples": len(rows),
                "elapsed_seconds_total": previous_elapsed + elapsed,
                "status": "selection_complete",
            },
        )
    print(f"IFEVAL GENERATION COMPLETE: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
