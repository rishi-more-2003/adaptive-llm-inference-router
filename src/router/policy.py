from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from src.router.features import RequestFeatures
from src.router.latency import HeuristicLatencyPredictor


class InferencePath(Enum):
    BASELINE = "baseline"
    VLLM = "vllm"
    INT4_GEMV = "int4_gemv"
    PREFIX_CACHE = "prefix_cache"
    SPECULATIVE = "speculative"
    LONG_CONTEXT = "long_context"


@dataclass(frozen=True)
class RouterConfig:
    short_prompt_threshold_tokens: int = 512
    short_decode_threshold_tokens: int = 256
    long_context_threshold_tokens: int = 4096
    tight_latency_target_ms: int = 1000
    prefix_cache_enabled: bool = True
    int4_gemv_enabled: bool = True
    speculative_enabled: bool = False
    vllm_enabled: bool = False
    vllm_available: bool = False
    reject_impossible_latency_targets: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any], vllm_available: bool = False) -> "RouterConfig":
        accepted = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in values.items() if key in accepted}
        filtered["vllm_available"] = vllm_available
        return cls(**filtered)


@dataclass
class RoutingDecision:
    path: InferencePath
    reason: str
    features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.value,
            "reason": self.reason,
            "features": self.features,
        }


class AdaptivePolicy:
    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        self.latency_predictor = HeuristicLatencyPredictor()

    def decide(self, features: RequestFeatures) -> RoutingDecision:
        feature_dict = asdict(features)
        latency_prediction = self.latency_predictor.predict(features)
        feature_dict["latency_prediction"] = latency_prediction.to_dict()
        feature_dict["admission_control"] = {
            "enabled": self.config.reject_impossible_latency_targets,
            "should_reject": bool(
                self.config.reject_impossible_latency_targets
                and latency_prediction.latency_target_feasible is False
            ),
            "recommendation": latency_prediction.admission_recommendation,
            "reason": latency_prediction.reason,
        }
        cfg = self.config

        if features.prefix_cache_hit and cfg.prefix_cache_enabled:
            return RoutingDecision(
                path=InferencePath.PREFIX_CACHE,
                reason="prefix cache hit; simulated prefix reuse can avoid repeated prompt processing",
                features=feature_dict,
            )

        if features.estimated_prompt_tokens > cfg.long_context_threshold_tokens:
            return RoutingDecision(
                path=InferencePath.LONG_CONTEXT,
                reason=(
                    f"prompt_tokens={features.estimated_prompt_tokens} exceeds "
                    f"long_context_threshold={cfg.long_context_threshold_tokens}; routing to KV-aware stub"
                ),
                features=feature_dict,
            )

        if features.request_type == "code":
            if cfg.vllm_enabled and cfg.vllm_available:
                return RoutingDecision(
                    path=InferencePath.VLLM,
                    reason="code prompt detected and vLLM routing is enabled",
                    features=feature_dict,
                )
            return RoutingDecision(
                path=InferencePath.BASELINE,
                reason="code prompt detected; using reliable baseline path",
                features=feature_dict,
            )

        if (
            features.batch_size == 1
            and features.estimated_prompt_tokens < cfg.short_prompt_threshold_tokens
            and features.max_new_tokens <= cfg.short_decode_threshold_tokens
            and cfg.int4_gemv_enabled
        ):
            return RoutingDecision(
                path=InferencePath.INT4_GEMV,
                reason=(
                    f"batch_size=1, prompt_tokens={features.estimated_prompt_tokens}, "
                    f"max_new_tokens={features.max_new_tokens}; expected memory-bandwidth-bound decode path"
                ),
                features=feature_dict,
            )

        if (
            features.latency_target_ms is not None
            and features.latency_target_ms <= cfg.tight_latency_target_ms
            and cfg.speculative_enabled
        ):
            return RoutingDecision(
                path=InferencePath.SPECULATIVE,
                reason=(
                    f"latency_target_ms={features.latency_target_ms} is tight and speculative routing is enabled"
                ),
                features=feature_dict,
            )

        if cfg.vllm_enabled and cfg.vllm_available:
            return RoutingDecision(
                path=InferencePath.VLLM,
                reason="vLLM routing enabled and vLLM is available",
                features=feature_dict,
            )

        return RoutingDecision(
            path=InferencePath.BASELINE,
            reason="no specialized routing rule matched; using reliable HuggingFace baseline",
            features=feature_dict,
        )
