#!/usr/bin/env python3
"""Pre-tokenize SFT JSONL on CPU so paid GPU time starts at model loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from position_bias_research.chat_protocol import apply_chat_template, selected_protocol_for_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def portable_source(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the exact tokenizer rules that turn conversations into token IDs."""
    payload = {
        "backend": tokenizer.backend_tokenizer.to_str(),
        "chat_template": tokenizer.chat_template,
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


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


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") not in {
                "position-sft-v1",
                "position-sft-v2",
            }:
                raise ValueError(f"Unexpected schema at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty input: {path}")
    return rows


def token_ids(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
    encoded = apply_chat_template(tokenizer, messages, tokenize=True, **kwargs)
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Expected a single tokenized conversation")
        encoded = encoded[0]
    return [int(value) for value in encoded]


def encode_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    messages = row["messages"]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"Unexpected roles in {row['id']}")
    prompt_ids = token_ids(
        tokenizer,
        messages[:-1],
        add_generation_prompt=True,
    )
    full_ids = token_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Chat template prefix mismatch in {row['id']}; cannot build a safe completion mask"
        )
    if len(full_ids) > max_length:
        raise ValueError(
            f"{row['id']} has {len(full_ids)} tokens, exceeding --max-length={max_length}; "
            "refusing silent truncation"
        )
    return {
        "input_ids": full_ids,
        "completion_mask": [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-length", type=int, default=8320)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-artifact-hashes",
        action="store_true",
        help="Refresh Arrow artifact hashes without re-tokenizing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.refresh_artifact_hashes:
        metadata_path = args.output / "pretokenization.json"
        if not metadata_path.is_file():
            raise SystemExit(f"Missing metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["artifact_files"] = artifact_manifest(args.output)
        if args.tokenizer_revision:
            metadata["tokenizer_revision"] = args.tokenizer_revision
        temporary = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.output}; pass --overwrite to replace it")
        shutil.rmtree(args.output)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    rows = read_rows(args.input)
    encoded: list[dict[str, Any]] = []
    lengths: list[int] = []
    completion_lengths: list[int] = []
    for index, row in enumerate(rows, start=1):
        item = encode_row(tokenizer, row, args.max_length)
        encoded.append(item)
        lengths.append(len(item["input_ids"]))
        completion_lengths.append(sum(item["completion_mask"]))
        if index % 100 == 0:
            print(f"tokenized {index}/{len(rows)}", flush=True)

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "completion_mask": Sequence(Value("int8")),
        }
    )
    dataset = Dataset.from_list(encoded, features=features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output))
    artifact_files = artifact_manifest(args.output)
    metadata = {
        "schema_version": "pretokenized-sft-v1",
        "source": portable_source(args.input),
        "source_sha256": sha256_file(args.input),
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "chat_protocol": selected_protocol_for_tokenizer(tokenizer),
        "rows": len(dataset),
        "max_length_limit": args.max_length,
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "total_tokens": sum(lengths),
        "min_completion_tokens": min(completion_lengths),
        "max_completion_tokens": max(completion_lengths),
        "total_completion_tokens": sum(completion_lengths),
        "artifact_files": artifact_files,
    }
    metadata_path = args.output / "pretokenization.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
