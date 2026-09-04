"""Token counting adapters used during CPU-only data preparation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .chat_protocol import apply_chat_template


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def count_chat(self, system_prompt: str, user_prompt: str) -> int: ...


@dataclass(frozen=True)
class WhitespaceTokenCounter:
    """Dependency-free counter for tests and smoke runs, not final experiments."""

    name: str = "whitespace-smoke-only"

    def count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    def count_chat(self, system_prompt: str, user_prompt: str) -> int:
        return self.count(f"{system_prompt}\n\n{user_prompt}")


class HuggingFaceTokenCounter:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face tokenization requires the optional data dependencies. "
                "Install them with: pip install -e '.[data]'"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
            local_files_only=local_files_only,
        )
        # Keep dataset condition labels stable across pinned/unpinned loading;
        # the exact revision and tokenizer fingerprint live in the model/data
        # manifests instead of changing every sample row.
        self.name = model_name

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def count_chat(self, system_prompt: str, user_prompt: str) -> int:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        token_ids = apply_chat_template(
            self._tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        # transformers 5 may return BatchEncoding, while 4.x returns a list.
        if hasattr(token_ids, "get"):
            input_ids = token_ids.get("input_ids")
            if input_ids is None:
                raise RuntimeError("chat template output does not contain input_ids")
            return len(input_ids)
        return len(token_ids)


def load_token_counter(
    tokenizer_name: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> TokenCounter:
    if tokenizer_name == "whitespace":
        return WhitespaceTokenCounter()
    return HuggingFaceTokenCounter(
        tokenizer_name,
        revision=revision,
        local_files_only=local_files_only,
    )
