from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.backends.base import GenerationResult, InferenceBackend
from src.backends.hf_backend import HuggingFaceBackend
from src.backends.int4_gemv_backend import Int4GemvBackend
from src.backends.long_context_backend import LongContextBackend
from src.backends.speculative_backend import SpeculativeBackend
from src.backends.vllm_backend import VLLMBackend
from src.cache.prefix_cache import PrefixCacheSimulator
from src.router.features import RequestFeatureExtractor, RequestFeatures
from src.router.policy import AdaptivePolicy, InferencePath, RouterConfig, RoutingDecision


@dataclass
class RoutedGeneration:
    result: GenerationResult
    decision: RoutingDecision
    prefix_cache_metrics: dict[str, float | int]


class AdaptiveInferenceRouter:
    def __init__(self, config: dict[str, Any]) -> None:
        model_cfg = config.get("model", {})
        router_cfg = config.get("router", {})
        cache_cfg = config.get("prefix_cache", {})
        vllm_cfg = config.get("vllm", {})
        speculative_cfg = config.get("speculative", {})

        self.prefix_cache = PrefixCacheSimulator(
            prefix_chars=int(cache_cfg.get("prefix_chars", 1024)),
            max_entries=int(cache_cfg.get("max_entries", 1000)),
        )
        self.feature_extractor = RequestFeatureExtractor(
            long_context_threshold_tokens=int(router_cfg.get("long_context_threshold_tokens", 4096))
        )

        self.baseline_backend = HuggingFaceBackend(
            model_name=str(model_cfg.get("name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")),
            device=str(model_cfg.get("device", "auto")),
            dtype=str(model_cfg.get("dtype", "auto")),
        )
        self.vllm_backend = VLLMBackend(
            model_name=str(model_cfg.get("name", self.baseline_backend.model_name)),
            server_url=vllm_cfg.get("server_url"),
            request_timeout_s=float(vllm_cfg.get("request_timeout_s", 600)),
        )
        self.int4_backend = Int4GemvBackend(self.baseline_backend)
        self.speculative_backend = SpeculativeBackend(
            self.baseline_backend,
            draft_model_name=speculative_cfg.get("draft_model_name"),
        )
        self.long_context_backend = LongContextBackend(self.baseline_backend)

        policy_config = RouterConfig.from_dict(router_cfg, vllm_available=self.vllm_backend.available)
        self.policy = AdaptivePolicy(policy_config)

    def inspect_route(
        self,
        prompt: str,
        max_tokens: int,
        latency_target_ms: int | None = None,
        batch_size: int = 1,
        update_prefix_cache: bool = False,
    ) -> RoutingDecision:
        prefix_result = None
        if update_prefix_cache:
            prefix_result = self.prefix_cache.check_and_update(prompt)
        features = self.feature_extractor.extract(
            prompt=prompt,
            max_new_tokens=max_tokens,
            latency_target_ms=latency_target_ms,
            batch_size=batch_size,
            prefix_cache_result=prefix_result,
        )
        return self.policy.decide(features)

    def _backend_for_path(self, path: InferencePath) -> InferenceBackend:
        if path == InferencePath.VLLM:
            return self.vllm_backend
        if path == InferencePath.INT4_GEMV:
            return self.int4_backend
        if path == InferencePath.SPECULATIVE:
            return self.speculative_backend
        if path == InferencePath.LONG_CONTEXT:
            return self.long_context_backend
        return self.baseline_backend

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        latency_target_ms: int | None = None,
        batch_size: int = 1,
    ) -> RoutedGeneration:
        prefix_result = self.prefix_cache.check_and_update(prompt)
        features: RequestFeatures = self.feature_extractor.extract(
            prompt=prompt,
            max_new_tokens=max_tokens,
            latency_target_ms=latency_target_ms,
            batch_size=batch_size,
            prefix_cache_result=prefix_result,
        )
        decision = self.policy.decide(features)
        backend = self._backend_for_path(decision.path)

        request_metadata: dict[str, Any] = {
            "routing_path": decision.path.value,
            "routing_reason": decision.reason,
            "prefix_cache_hit": prefix_result.hit,
            "prefix_cache_hash": prefix_result.prefix_hash,
            "prefix_cache_tokens_saved": prefix_result.estimated_tokens_saved,
        }
        if decision.path == InferencePath.PREFIX_CACHE:
            request_metadata["simulated_path"] = "PREFIX_CACHE_SIMULATED"
            request_metadata["honesty_note"] = (
                "Prefix reuse is simulated in v1; generation still uses the fallback backend."
            )

        result = backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            request_metadata=request_metadata,
        )
        return RoutedGeneration(
            result=result,
            decision=decision,
            prefix_cache_metrics=self.prefix_cache.metrics(),
        )
