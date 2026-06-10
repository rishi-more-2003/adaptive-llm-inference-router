from __future__ import annotations

from dataclasses import asdict, dataclass

from src.cache.prefix_cache import PrefixCacheResult
from src.utils.gpu import get_gpu_stats
from src.utils.tokens import estimate_tokens


@dataclass
class RequestFeatures:
    prompt_chars: int
    estimated_prompt_tokens: int
    max_new_tokens: int
    batch_size: int
    latency_target_ms: int | None
    has_shared_prefix: bool
    prefix_cache_hit: bool
    estimated_kv_cache_mb: float
    request_type: str
    cuda_available: bool
    gpu_memory_allocated_mb: float | None
    gpu_memory_reserved_mb: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_request(prompt: str, estimated_prompt_tokens: int, long_context_threshold: int) -> str:
    lower = prompt.lower()
    if estimated_prompt_tokens > long_context_threshold:
        return "long_context"
    if "```" in prompt or "write code" in lower or "implement " in lower:
        return "code"
    if "context:" in lower or "documents:" in lower:
        return "rag"
    return "chat"


def estimate_kv_cache_mb(
    estimated_prompt_tokens: int,
    max_new_tokens: int,
    batch_size: int = 1,
    hidden_size: int = 2048,
    num_layers: int = 22,
    bytes_per_value: int = 2,
) -> float:
    total_tokens = estimated_prompt_tokens + max_new_tokens
    # KV cache stores both key and value tensors per layer.
    bytes_used = batch_size * total_tokens * num_layers * hidden_size * 2 * bytes_per_value
    return bytes_used / (1024 * 1024)


class RequestFeatureExtractor:
    def __init__(self, long_context_threshold_tokens: int = 4096) -> None:
        self.long_context_threshold_tokens = long_context_threshold_tokens

    def extract(
        self,
        prompt: str,
        max_new_tokens: int,
        latency_target_ms: int | None = None,
        batch_size: int = 1,
        prefix_cache_result: PrefixCacheResult | None = None,
    ) -> RequestFeatures:
        estimated_prompt_tokens = estimate_tokens(prompt)
        gpu_stats = get_gpu_stats()
        prefix_hit = bool(prefix_cache_result.hit) if prefix_cache_result else False
        has_shared_prefix = prefix_hit or bool(prefix_cache_result and prefix_cache_result.estimated_tokens_saved > 0)
        request_type = classify_request(
            prompt,
            estimated_prompt_tokens,
            self.long_context_threshold_tokens,
        )

        return RequestFeatures(
            prompt_chars=len(prompt),
            estimated_prompt_tokens=estimated_prompt_tokens,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            latency_target_ms=latency_target_ms,
            has_shared_prefix=has_shared_prefix,
            prefix_cache_hit=prefix_hit,
            estimated_kv_cache_mb=estimate_kv_cache_mb(
                estimated_prompt_tokens,
                max_new_tokens,
                batch_size=batch_size,
            ),
            request_type=request_type,
            cuda_available=gpu_stats.cuda_available,
            gpu_memory_allocated_mb=gpu_stats.memory_allocated_mb,
            gpu_memory_reserved_mb=gpu_stats.memory_reserved_mb,
        )
