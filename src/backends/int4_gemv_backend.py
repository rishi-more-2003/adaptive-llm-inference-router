from __future__ import annotations

import logging
from typing import Any

from src.backends.base import GenerationResult, InferenceBackend

LOGGER = logging.getLogger(__name__)


class Int4GemvBackend(InferenceBackend):
    name = "int4_gemv_stub"

    def __init__(self, fallback: InferenceBackend) -> None:
        self.fallback = fallback

    def decode_one_token_int4(self, hidden_state: Any, quantized_weights: Any, scales: Any) -> Any:
        """
        Future extension point for fused INT4 dequant + GEMV decode.
        """
        raise NotImplementedError("INT4 GEMV CUDA/Triton kernel is not implemented in v1.")

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        LOGGER.info("INT4_GEMV_STUB selected; delegating generation to %s", self.fallback.name)
        metadata = {
            **(request_metadata or {}),
            "stub_path": "INT4_GEMV_STUB",
            "honesty_note": "No custom INT4 kernel is implemented in v1; using fallback backend.",
        }
        result = self.fallback.generate(prompt, max_tokens, temperature, metadata)
        result.backend_name = self.name
        result.metadata["underlying_backend"] = self.fallback.name
        return result
