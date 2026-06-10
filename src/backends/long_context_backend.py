from __future__ import annotations

import logging
from typing import Any

from src.backends.base import GenerationResult, InferenceBackend

LOGGER = logging.getLogger(__name__)


class LongContextBackend(InferenceBackend):
    name = "long_context_stub"

    def __init__(self, fallback: InferenceBackend) -> None:
        self.fallback = fallback

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        LOGGER.info("LONG_CONTEXT_STUB selected; delegating generation to %s", self.fallback.name)
        metadata = {
            **(request_metadata or {}),
            "stub_path": "LONG_CONTEXT_STUB",
            "honesty_note": (
                "Paged/FP8 KV-cache long-context optimization is not implemented in v1; "
                "using fallback backend."
            ),
        }
        result = self.fallback.generate(prompt, max_tokens, temperature, metadata)
        result.backend_name = self.name
        result.metadata["underlying_backend"] = self.fallback.name
        return result
