from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    total_latency_ms: float
    ttft_ms: float | None
    tokens_per_sec: float
    backend_name: str
    metadata: dict[str, Any]


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        pass
