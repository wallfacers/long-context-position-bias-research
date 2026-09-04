"""Deterministic chat rendering compatibility shared by data, training, and eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


NATIVE = "native-system-user-assistant"
MERGE_SYSTEM = "merge-system-into-first-user-v1"
SUPPORTED = {NATIVE, MERGE_SYSTEM}


def normalize_messages(
    messages: Sequence[dict[str, str]], protocol: str
) -> list[dict[str, str]]:
    copied = [dict(message) for message in messages]
    if protocol == NATIVE:
        return copied
    if protocol != MERGE_SYSTEM:
        raise ValueError(f"Unsupported chat protocol: {protocol}")
    if not copied or copied[0].get("role") != "system":
        return copied
    if len(copied) < 2 or copied[1].get("role") != "user":
        raise ValueError("A leading system message must be followed by a user message")
    merged = {
        "role": "user",
        "content": copied[0]["content"] + "\n\n" + copied[1]["content"],
    }
    return [merged, *copied[2:]]


def selected_protocol_for_tokenizer(tokenizer: Any) -> str:
    cached = getattr(tokenizer, "_position_bias_chat_protocol", None)
    if cached in SUPPORTED:
        return cached
    name = str(getattr(tokenizer, "name_or_path", ""))
    audit_path = Path(name) / "chat_protocol_audit.json"
    protocol = NATIVE
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "passed":
            raise ValueError(f"Chat protocol audit has not passed: {audit_path}")
        protocol = str(audit.get("selected_protocol", audit.get("protocol", NATIVE)))
        if protocol not in SUPPORTED:
            raise ValueError(f"Unknown selected chat protocol in {audit_path}: {protocol}")
    setattr(tokenizer, "_position_bias_chat_protocol", protocol)
    return protocol


def apply_chat_template(tokenizer: Any, messages: Sequence[dict[str, str]], **kwargs: Any) -> Any:
    protocol = selected_protocol_for_tokenizer(tokenizer)
    return tokenizer.apply_chat_template(normalize_messages(messages, protocol), **kwargs)
