#!/usr/bin/env python3
"""Audit that a tokenizer's native chat template preserves the frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from position_bias_research.chat_protocol import MERGE_SYSTEM, NATIVE, normalize_messages


MESSAGES = [
    {"role": "system", "content": "SYSTEM_SENTINEL_7f31"},
    {"role": "user", "content": "USER_SENTINEL_0c92"},
    {"role": "assistant", "content": "ASSISTANT_SENTINEL_98ab"},
]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Expected one conversation from apply_chat_template")
        encoded = encoded[0]
    return [int(value) for value in encoded]


def audit_protocol(tokenizer: Any, protocol: str) -> dict[str, Any]:
    messages = normalize_messages(MESSAGES, protocol)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = normalize_ids(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = normalize_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    )
    sentinels = {
        "system": MESSAGES[0]["content"] in prompt_text,
        "user": MESSAGES[1]["content"] in prompt_text,
        "assistant": MESSAGES[2]["content"] in full_text,
    }
    prefix_safe = full_ids[: len(prompt_ids)] == prompt_ids
    status = "passed" if all(sentinels.values()) and prefix_safe else "failed"
    return {
        "schema_version": "chat-protocol-audit-v1",
        "status": status,
        "protocol": protocol,
        "sentinels_preserved": sentinels,
        "completion_mask_prefix_safe": prefix_safe,
        "prompt_tokens": len(prompt_ids),
        "full_tokens": len(full_ids),
        "completion_tokens": len(full_ids) - len(prompt_ids) if prefix_safe else None,
        "prompt_render_sha256": sha256_text(prompt_text),
        "full_render_sha256": sha256_text(full_text),
        "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
    }


def audit_tokenizer(tokenizer: Any) -> dict[str, Any]:
    native = audit_protocol(tokenizer, NATIVE)
    if native["status"] == "passed":
        return native | {
            "selected_protocol": NATIVE,
            "native_protocol_status": "passed",
        }
    merged = audit_protocol(tokenizer, MERGE_SYSTEM)
    if merged["status"] == "passed":
        return merged | {
            "selected_protocol": MERGE_SYSTEM,
            "native_protocol_status": "failed",
            "native_completion_mask_prefix_safe": native[
                "completion_mask_prefix_safe"
            ],
            "compatibility_rationale": (
                "The native full-conversation rendering differs from inference rendering; "
                "merge the leading system text into the first user turn for both paths."
            ),
        }
    return merged | {
        "status": "failed",
        "selected_protocol": None,
        "native_protocol_status": "failed",
        "native_completion_mask_prefix_safe": native[
            "completion_mask_prefix_safe"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    try:
        payload = audit_tokenizer(tokenizer)
    except Exception as exc:  # preserve a machine-readable failure for the audit trail
        payload = {
            "schema_version": "chat-protocol-audit-v1",
            "status": "failed",
            "protocol": "auto-native-or-merge-system",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    payload.update(
        {
            "tokenizer": args.tokenizer,
            "revision": args.revision,
            "local_files_only": args.local_files_only,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
