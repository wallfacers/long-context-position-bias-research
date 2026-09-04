#!/usr/bin/env python3
"""Build a pinned experiment config from a complete local model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SIGNATURE_KEYS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    payload = {
        "backend": tokenizer.backend_tokenizer.to_str(),
        "chat_template": tokenizer.chat_template,
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def config_signature(config: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in SIGNATURE_KEYS if config.get(key) is None]
    if missing:
        raise ValueError("Model config lacks signature keys: " + ", ".join(missing))
    return {key: config[key] for key in SIGNATURE_KEYS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-model-length", type=int, required=True)
    parser.add_argument("--training-max-length", type=int, default=8320)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("--revision must be a pinned 40-character lowercase git commit")
    if args.training_max_length > args.max_model_length:
        raise SystemExit("Training length cannot exceed the registered model length")
    config_path = args.model / "config.json"
    tokenizer_path = args.model / "tokenizer_config.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise SystemExit("A complete local model/tokenizer snapshot is required")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    actual = json.loads(config_path.read_text(encoding="utf-8"))
    chat_audit_path = args.model / "chat_protocol_audit.json"
    if not chat_audit_path.is_file():
        raise SystemExit("Run audit_chat_protocol.py before pinning the model config")
    chat_audit = json.loads(chat_audit_path.read_text(encoding="utf-8"))
    if chat_audit.get("status") != "passed" or not chat_audit.get("selected_protocol"):
        raise SystemExit("The model chat-protocol audit has not passed")
    registered_window = int(actual.get("max_position_embeddings", 0))
    if registered_window and args.max_model_length > registered_window:
        raise SystemExit(
            f"Requested window {args.max_model_length} exceeds config max_position_embeddings "
            f"{registered_window}"
        )
    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "chat_protocol": chat_audit["selected_protocol"],
        "chat_protocol_audit_sha256": hashlib.sha256(
            chat_audit_path.read_bytes()
        ).hexdigest(),
        "max_model_length": args.max_model_length,
        "training_max_length": args.training_max_length,
        "config_signature": config_signature(actual),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
