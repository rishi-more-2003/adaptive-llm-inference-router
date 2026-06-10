from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

from src.utils.tokens import estimate_tokens, normalize_text


@dataclass(frozen=True)
class PrefixCacheResult:
    hit: bool
    prefix_hash: str
    estimated_tokens_saved: int


class PrefixCacheSimulator:
    def __init__(self, prefix_chars: int = 1024, max_entries: int = 1000) -> None:
        self.prefix_chars = prefix_chars
        self.max_entries = max_entries
        self._entries: OrderedDict[str, int] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.estimated_tokens_saved = 0

    def _hash_prefix(self, prompt: str) -> tuple[str, int]:
        normalized = normalize_text(prompt)
        prefix = normalized[: self.prefix_chars]
        prefix_hash = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        return prefix_hash, estimate_tokens(prefix)

    def check_and_update(self, prompt: str) -> PrefixCacheResult:
        prefix_hash, token_estimate = self._hash_prefix(prompt)
        hit = prefix_hash in self._entries

        if hit:
            self.hits += 1
            self.estimated_tokens_saved += token_estimate
            self._entries.move_to_end(prefix_hash)
            saved = token_estimate
        else:
            self.misses += 1
            self._entries[prefix_hash] = token_estimate
            saved = 0
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

        return PrefixCacheResult(hit=hit, prefix_hash=prefix_hash, estimated_tokens_saved=saved)

    def metrics(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        return {
            "prefix_cache_hits": self.hits,
            "prefix_cache_misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "entries": len(self._entries),
        }
