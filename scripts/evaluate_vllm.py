#!/usr/bin/env python3
"""Resumable offline vLLM evaluation with constrained JSON responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from position_bias_research.chat_protocol import apply_chat_template


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_quotes": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["answer", "evidence_ids", "evidence_quotes", "confidence"],
    "additionalProperties": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Empty adapter directory: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "position-group-v1":
                raise ValueError(f"Unexpected schema at {path}:{line_number}")
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    return rows


def existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid existing result at {path}:{line_number}; preserve the file and repair the final line"
                ) from exc
            ids.add(row["sample_id"])
    return ids


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["prompt"]},
    ]
    return apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_english_answer(value: str) -> str:
    """LongBench/SQuAD-style normalization for natural QA scoring."""

    lowered = str(value).lower()
    without_punctuation = "".join(
        character for character in lowered if character not in set(string.punctuation)
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_english_answer(prediction).split()
    expected = normalize_english_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    shared = Counter(predicted) & Counter(expected)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def answer_score(parsed: dict[str, Any], target: dict[str, Any], metadata: dict[str, Any]) -> float:
    prediction = str(parsed.get("answer", ""))
    references = [str(value) for value in target.get("answers", [target["answer"]])]
    metric = metadata.get("answer_metric", "exact")
    if metric == "exact":
        return float(any(prediction == reference for reference in references))
    if metric == "normalized_exact":
        normalized = normalize_english_answer(prediction)
        return float(any(normalized == normalize_english_answer(reference) for reference in references))
    if metric == "qa_f1_en":
        return max(token_f1(prediction, reference) for reference in references)
    raise ValueError(f"Unsupported answer metric: {metric}")


def score(row: dict[str, Any], generated: str) -> dict[str, Any]:
    try:
        parsed = json.loads(generated)
        valid_json = isinstance(parsed, dict)
    except json.JSONDecodeError:
        parsed = {}
        valid_json = False
    target = row["target"]
    predicted_ids = set(parsed.get("evidence_ids", [])) if valid_json else set()
    predicted_quotes = set(parsed.get("evidence_quotes", [])) if valid_json else set()
    expected_ids = set(target["evidence_ids"])
    expected_quotes = set(target["evidence_quotes"])
    metadata = row.get("metadata", {})
    numeric_answer_score = answer_score(parsed, target, metadata) if valid_json else 0.0
    evidence_ids_applicable = bool(
        metadata.get("evidence_id_applicable", bool(expected_ids))
    )
    evidence_quotes_applicable = bool(
        metadata.get("evidence_quote_applicable", bool(expected_quotes))
    )
    return {
        "valid_json": valid_json,
        "answer_metric": metadata.get("answer_metric", "exact"),
        "answer_score": numeric_answer_score,
        "answer_correct": valid_json and numeric_answer_score == 1.0,
        "evidence_ids_applicable": evidence_ids_applicable,
        "evidence_ids_correct": (
            valid_json and expected_ids.issubset(predicted_ids)
            if evidence_ids_applicable
            else None
        ),
        "evidence_quotes_applicable": evidence_quotes_applicable,
        "evidence_quotes_correct": (
            valid_json and expected_quotes.issubset(predicted_quotes)
            if evidence_quotes_applicable
            else None
        ),
        "all_predicted_quotes_supported_applicable": evidence_quotes_applicable,
        "all_predicted_quotes_supported": (
            valid_json
            and bool(predicted_quotes)
            and all(quote in row["prompt"] for quote in predicted_quotes)
            if evidence_quotes_applicable
            else None
        ),
        "parsed": parsed if valid_json else None,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filters = {
        "task": set(args.task or []),
        "filler_type": set(args.filler or []),
        "target_tokens": set(args.length or []),
        "position_label": set(args.position or []),
    }
    selected = [
        row
        for row in rows
        if all(not allowed or row[field] in allowed for field, allowed in filters.items())
    ]
    if args.max_samples is not None:
        if args.spread_samples and args.max_samples < len(selected):
            size = len(selected)
            selected = [
                selected[(index * size) // args.max_samples]
                for index in range(args.max_samples)
            ]
        else:
            selected = selected[: args.max_samples]
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--spread-samples",
        action="store_true",
        help="Spread --max-samples evenly across conditions instead of taking a prefix.",
    )
    parser.add_argument("--task", action="append", choices=("kv", "two_hop"))
    parser.add_argument("--filler", action="append")
    parser.add_argument("--length", action="append", type=int)
    parser.add_argument("--position", action="append")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.model.is_dir():
        raise SystemExit("--model must be a complete local directory")
    if args.adapter and not args.adapter.is_dir():
        raise SystemExit(f"Missing adapter: {args.adapter}")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    rows = filter_rows(load_rows(args.data), args)
    if not rows:
        raise SystemExit("No rows match the requested filters")
    completed = existing_ids(args.output)
    pending = [row for row in rows if row["sample_id"] not in completed]

    identity = {
        "schema_version": "vllm-eval-run-v1",
        "run_name": args.run_name,
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "adapter_sha256": sha256_directory(args.adapter) if args.adapter else None,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".run.json")
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_identity = {key: previous.get(key) for key in identity}
        if previous_identity != identity:
            raise SystemExit(
                f"Existing run metadata differs from this invocation: {metadata_path}"
            )
    else:
        write_json_atomic(metadata_path, identity | {"started_at": utc_now()})

    print(
        f"selected={len(rows)} completed={len(completed & {row['sample_id'] for row in rows})} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        print("Evaluation already complete for this selection; no GPU work needed.")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        local_files_only=True,
        trust_remote_code=False,
    )
    llm_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tokenizer": str(args.model),
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest

        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        lora_request = LoRARequest(args.run_name, 1, str(args.adapter))
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        structured_outputs=StructuredOutputsParams(json=RESPONSE_SCHEMA),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    processed = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for batch in batched(pending, args.batch_size):
            prompts = [build_prompt(tokenizer, row) for row in batch]
            request_started = time.monotonic()
            outputs = llm.generate(
                prompts,
                sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )
            batch_seconds = time.monotonic() - request_started
            if len(outputs) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} outputs, got {len(outputs)}")
            for row, output in zip(batch, outputs, strict=True):
                candidate = output.outputs[0]
                generated = candidate.text
                result = {
                    "schema_version": "position-eval-result-v1",
                    "run_name": args.run_name,
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
                f"saved={processed}/{len(pending)} elapsed={elapsed:.1f}s "
                f"sec/sample={elapsed / processed:.3f}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    finished = identity | {
        "started_at": json.loads(metadata_path.read_text(encoding="utf-8"))["started_at"],
        "last_finished_at": utc_now(),
        "samples_this_invocation": processed,
        "elapsed_seconds_this_invocation": elapsed,
        "seconds_per_sample_this_invocation": elapsed / processed,
        "status": "selection_complete",
    }
    write_json_atomic(metadata_path, finished)
    print(f"EVALUATION COMPLETE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
