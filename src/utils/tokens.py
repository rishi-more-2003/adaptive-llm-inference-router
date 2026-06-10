from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used before a real tokenizer is loaded."""
    stripped = text.strip()
    if not stripped:
        return 0

    wordish = len(re.findall(r"\S+", stripped))
    char_estimate = max(1, len(stripped) // 4)
    return max(1, max(wordish, char_estimate))


def normalize_text(text: str) -> str:
    """Normalize whitespace for prefix matching."""
    return " ".join(text.strip().split())
