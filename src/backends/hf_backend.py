from __future__ import annotations

import time
from typing import Any

from src.backends.base import GenerationResult, InferenceBackend


class HuggingFaceBackend(InferenceBackend):
    name = "hf_baseline"

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device_config = device
        self.dtype_config = dtype
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None

    def _resolve_dtype(self) -> Any:
        torch = self._torch
        if self.dtype_config != "auto":
            return getattr(torch, self.dtype_config)
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "HuggingFace backend requires torch and transformers. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self._torch = torch
        if self.device_config == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device_config

        dtype = self._resolve_dtype()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self._device == "cuda":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        if self._device != "cuda":
            self._model.to(self._device)
        self._model.eval()

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        self._load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        max_context_tokens = int(getattr(self._model.config, "max_position_embeddings", 2048))
        max_input_tokens = max(1, max_context_tokens - max_tokens)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        target_device = getattr(self._model, "device", self._device)
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        prompt_was_truncated = prompt_tokens >= max_input_tokens

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        start = time.perf_counter()
        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)
        total_latency_ms = (time.perf_counter() - start) * 1000

        generated_ids = output_ids[0][prompt_tokens:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = int(generated_ids.shape[-1])
        tokens_per_sec = generated_tokens / max(total_latency_ms / 1000, 1e-9)
        ttft_ms = total_latency_ms / max(generated_tokens, 1)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_latency_ms=total_latency_ms,
            ttft_ms=ttft_ms,
            tokens_per_sec=tokens_per_sec,
            backend_name=self.name,
            metadata={
                "model_name": self.model_name,
                "device": self._device,
                "dtype": str(self._resolve_dtype()),
                "approximate_ttft": True,
                "prompt_truncated": prompt_was_truncated,
                "max_context_tokens": max_context_tokens,
                **(request_metadata or {}),
            },
        )
