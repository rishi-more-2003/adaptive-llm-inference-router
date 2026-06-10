from __future__ import annotations

from dataclasses import dataclass

from src.router.features import RequestFeatures


@dataclass(frozen=True)
class LatencyPrediction:
    estimated_latency_ms: float
    estimated_prefill_ms: float
    estimated_decode_ms: float
    latency_target_ms: int | None
    latency_target_feasible: bool | None
    admission_recommendation: str
    reason: str

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return {
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_prefill_ms": self.estimated_prefill_ms,
            "estimated_decode_ms": self.estimated_decode_ms,
            "latency_target_ms": self.latency_target_ms,
            "latency_target_feasible": self.latency_target_feasible,
            "admission_recommendation": self.admission_recommendation,
            "reason": self.reason,
        }


class HeuristicLatencyPredictor:
    """Small transparent predictor used for routing/admission metadata."""

    def __init__(
        self,
        prefill_ms_per_token: float = 0.18,
        decode_ms_per_token: float = 42.0,
        gpu_speedup: float = 0.55,
        kv_pressure_threshold_mb: float = 1024.0,
    ) -> None:
        self.prefill_ms_per_token = prefill_ms_per_token
        self.decode_ms_per_token = decode_ms_per_token
        self.gpu_speedup = gpu_speedup
        self.kv_pressure_threshold_mb = kv_pressure_threshold_mb

    def predict(self, features: RequestFeatures) -> LatencyPrediction:
        prefill_ms = features.estimated_prompt_tokens * self.prefill_ms_per_token
        decode_ms = features.max_new_tokens * self.decode_ms_per_token
        if features.cuda_available:
            prefill_ms *= self.gpu_speedup
            decode_ms *= self.gpu_speedup

        batch_penalty = max(1.0, 1.0 + 0.18 * (features.batch_size - 1))
        kv_penalty = 1.0
        if features.estimated_kv_cache_mb > self.kv_pressure_threshold_mb:
            excess = features.estimated_kv_cache_mb / self.kv_pressure_threshold_mb
            kv_penalty += min(1.0, 0.15 * excess)

        estimated_latency_ms = (prefill_ms + decode_ms) * batch_penalty * kv_penalty
        feasible = None
        recommendation = "no_latency_target"
        reason = "no latency target was provided"

        if features.latency_target_ms is not None:
            feasible = estimated_latency_ms <= features.latency_target_ms
            if feasible:
                recommendation = "admit"
                reason = "estimated latency is within the requested target"
            else:
                recommendation = "warn"
                reason = "estimated latency exceeds the requested target"

        return LatencyPrediction(
            estimated_latency_ms=round(estimated_latency_ms, 3),
            estimated_prefill_ms=round(prefill_ms * batch_penalty * kv_penalty, 3),
            estimated_decode_ms=round(decode_ms * batch_penalty * kv_penalty, 3),
            latency_target_ms=features.latency_target_ms,
            latency_target_feasible=feasible,
            admission_recommendation=recommendation,
            reason=reason,
        )
