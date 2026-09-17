"""Token-aware context budgeting for the Groq GPT-OSS pipeline."""

from __future__ import annotations

import tiktoken

GPT_OSS_ENCODING = "o200k_harmony"
_ENCODING = tiktoken.get_encoding(GPT_OSS_ENCODING)


def count_tokens(text: str) -> int:
    """Return the approximate GPT-OSS token count for text."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Return text capped to ``max_tokens`` without splitting encoded tokens."""
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    encoded = _ENCODING.encode(text, disallowed_special=())
    if len(encoded) <= max_tokens:
        return text
    return _ENCODING.decode(encoded[:max_tokens]).rstrip()
