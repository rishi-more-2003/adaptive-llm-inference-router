from __future__ import annotations

import time
from typing import Any

import httpx

from src.backends.base import GenerationResult, InferenceBackend


class VLLMBackend(InferenceBackend):
    name = "vllm"

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        server_url: str | None = None,
        request_timeout_s: float = 600.0,
    ) -> None:
        self.model_name = model_name
        self.server_url = server_url.rstrip("/") if server_url else None
        self.request_timeout_s = request_timeout_s
        self._llm: Any | None = None
        self._sampling_params_cls: Any | None = None
        self.available, self.availability_error = self.check_availability(self.server_url)

    @staticmethod
    def is_available() -> bool:
        available, _ = VLLMBackend.check_availability()
        return available

    @staticmethod
    def check_availability(server_url: str | None = None) -> tuple[bool, str | None]:
        if server_url:
            try:
                response = httpx.get(f"{server_url.rstrip('/')}/health", timeout=2.0)
                response.raise_for_status()
            except Exception as exc:
                return False, f"external vLLM server unavailable at {server_url}: {exc}"
            return True, None

        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except Exception as exc:
            return False, str(exc)
        return True, None

    def _load(self) -> None:
        if self.server_url:
            return
        if self._llm is not None:
            return
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            detail = self.availability_error or str(exc)
            raise RuntimeError(
                "vLLM backend was selected, but vLLM is not usable in this environment. "
                f"Import error: {detail}. Install a compatible vLLM build separately with `pip install vllm`."
            ) from exc
        self._llm = LLM(model=self.model_name)
        self._sampling_params_cls = SamplingParams

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if self.server_url:
            return self._generate_external(prompt, max_tokens, temperature, request_metadata)

        self._load()
        assert self._llm is not None
        assert self._sampling_params_cls is not None

        start = time.perf_counter()
        sampling_params = self._sampling_params_cls(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate([prompt], sampling_params)
        total_latency_ms = (time.perf_counter() - start) * 1000
        output = outputs[0].outputs[0]
        generated_tokens = len(getattr(output, "token_ids", []) or [])
        prompt_tokens = len(getattr(outputs[0], "prompt_token_ids", []) or [])
        tokens_per_sec = generated_tokens / max(total_latency_ms / 1000, 1e-9)

        return GenerationResult(
            text=output.text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_latency_ms=total_latency_ms,
            ttft_ms=total_latency_ms / max(generated_tokens, 1),
            tokens_per_sec=tokens_per_sec,
            backend_name=self.name,
            metadata={"model_name": self.model_name, "mode": "native", **(request_metadata or {})},
        )

    def _generate_external(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if not self.server_url:
            raise RuntimeError("External vLLM server URL is not configured.")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        start = time.perf_counter()
        response = httpx.post(
            f"{self.server_url}/v1/completions",
            json=payload,
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()
        total_latency_ms = (time.perf_counter() - start) * 1000
        data = response.json()
        choice = data.get("choices", [{}])[0]
        usage = data.get("usage") or {}
        generated_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        tokens_per_sec = generated_tokens / max(total_latency_ms / 1000, 1e-9)

        return GenerationResult(
            text=choice.get("text", ""),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_latency_ms=total_latency_ms,
            ttft_ms=total_latency_ms / max(generated_tokens, 1),
            tokens_per_sec=tokens_per_sec,
            backend_name=self.name,
            metadata={
                "model_name": self.model_name,
                "mode": "external_openai_compatible",
                "server_url": self.server_url,
                **(request_metadata or {}),
            },
        )
