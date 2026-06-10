from __future__ import annotations

import logging
from typing import Any

from src.backends.base import GenerationResult, InferenceBackend

LOGGER = logging.getLogger(__name__)


class SpeculativeBackend(InferenceBackend):
    name = "speculative_stub"

    def __init__(self, fallback: InferenceBackend, draft_model_name: str | None = None) -> None:
        self.fallback = fallback
        self.draft_model_name = draft_model_name

    def generate_with_draft_model(
        self,
        prompt: str,
        draft_model_name: str,
        target_model_name: str,
        max_tokens: int,
    ) -> GenerationResult:
        raise NotImplementedError("Draft-model speculative decoding is not implemented in v1.")

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        LOGGER.info("SPECULATIVE_STUB selected; delegating generation to %s", self.fallback.name)
        metadata = {
            **(request_metadata or {}),
            "stub_path": "SPECULATIVE_STUB",
            "draft_model_name": self.draft_model_name,
            "speculative_mode": "configured_draft_model" if self.draft_model_name else "fallback_stub",
            "honesty_note": (
                "Draft-model speculative decoding requires a compatible draft model. "
                "No validated draft model is configured for this run; using fallback backend."
                if not self.draft_model_name
                else "A draft model is configured, but the v1 wrapper still delegates to fallback generation."
            ),
        }
        result = self.fallback.generate(prompt, max_tokens, temperature, metadata)
        result.backend_name = self.name
        result.metadata["underlying_backend"] = self.fallback.name
        return result
