from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RequestMetric:
    request_id: str
    timestamp: float
    model: str
    selected_inference_path: str
    routing_reason: str
    backend_used: str
    prompt_tokens: int
    generated_tokens: int
    total_latency_ms: float
    ttft_ms: float | None
    inter_token_latency_ms: float | None
    tokens_per_sec: float
    latency_target_ms: int | None
    latency_target_met: bool | None
    prefix_cache_hit: bool
    estimated_kv_cache_mb: float
    gpu_memory_allocated_mb: float | None
    gpu_memory_reserved_mb: float | None


class MetricsCollector:
    def __init__(self, output_path: str | Path = "runs/metrics.jsonl") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[RequestMetric] = []

    def record(self, metric: RequestMetric) -> None:
        self.records.append(metric)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(metric)) + "\n")

    def aggregate(self, prefix_cache_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.records:
            return {
                "requests": 0,
                "p50_latency_ms": None,
                "p95_latency_ms": None,
                "avg_tokens_per_sec": None,
                "latency_target_attainment": None,
                "path_counts": {},
                "prefix_cache_hit_rate": 0.0,
                "prefix_cache": prefix_cache_metrics or {},
            }

        latencies = sorted(record.total_latency_ms for record in self.records)
        path_counts = Counter(record.selected_inference_path for record in self.records)
        targets = [record for record in self.records if record.latency_target_met is not None]
        target_attainment = (
            sum(1 for record in targets if record.latency_target_met) / len(targets) if targets else None
        )
        prefix_hits = sum(1 for record in self.records if record.prefix_cache_hit)

        return {
            "requests": len(self.records),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "avg_tokens_per_sec": statistics.mean(record.tokens_per_sec for record in self.records),
            "latency_target_attainment": target_attainment,
            "path_counts": dict(path_counts),
            "prefix_cache_hit_rate": prefix_hits / len(self.records),
            "prefix_cache": prefix_cache_metrics or {},
        }

    def prometheus_text(self, prefix_cache_metrics: dict[str, Any] | None = None) -> str:
        aggregate = self.aggregate(prefix_cache_metrics)
        lines = [
            "# HELP adaptive_router_requests_total Total completed generation requests.",
            "# TYPE adaptive_router_requests_total counter",
            f"adaptive_router_requests_total {aggregate['requests']}",
            "# HELP adaptive_router_latency_ms Request latency percentiles in milliseconds.",
            "# TYPE adaptive_router_latency_ms gauge",
            f"adaptive_router_latency_ms{{quantile=\"0.50\"}} {_prom_value(aggregate['p50_latency_ms'])}",
            f"adaptive_router_latency_ms{{quantile=\"0.95\"}} {_prom_value(aggregate['p95_latency_ms'])}",
            "# HELP adaptive_router_tokens_per_second Average generation throughput.",
            "# TYPE adaptive_router_tokens_per_second gauge",
            f"adaptive_router_tokens_per_second {_prom_value(aggregate['avg_tokens_per_sec'])}",
            "# HELP adaptive_router_latency_target_attainment Ratio of requests meeting latency target.",
            "# TYPE adaptive_router_latency_target_attainment gauge",
            f"adaptive_router_latency_target_attainment {_prom_value(aggregate['latency_target_attainment'])}",
            "# HELP adaptive_router_prefix_cache_hit_rate Prefix cache hit rate.",
            "# TYPE adaptive_router_prefix_cache_hit_rate gauge",
            f"adaptive_router_prefix_cache_hit_rate {aggregate['prefix_cache_hit_rate']}",
            "# HELP adaptive_router_path_requests_total Completed requests by selected inference path.",
            "# TYPE adaptive_router_path_requests_total counter",
        ]
        for path, count in sorted(aggregate["path_counts"].items()):
            lines.append(f'adaptive_router_path_requests_total{{path="{path}"}} {count}')

        prefix_metrics = aggregate.get("prefix_cache") or {}
        for key in ("prefix_cache_hits", "prefix_cache_misses", "estimated_tokens_saved", "entries"):
            if key in prefix_metrics:
                lines.append(f"adaptive_router_{key} {prefix_metrics[key]}")

        return "\n".join(lines) + "\n"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile)))
    return values[index]


def _prom_value(value: Any) -> float | str:
    return value if value is not None else "NaN"


def build_request_metric(
    request_id: str,
    model: str,
    selected_inference_path: str,
    routing_reason: str,
    backend_used: str,
    prompt_tokens: int,
    generated_tokens: int,
    total_latency_ms: float,
    ttft_ms: float | None,
    tokens_per_sec: float,
    latency_target_ms: int | None,
    prefix_cache_hit: bool,
    features: dict[str, Any],
) -> RequestMetric:
    inter_token_latency_ms = total_latency_ms / generated_tokens if generated_tokens else None
    latency_target_met = (
        total_latency_ms <= latency_target_ms if latency_target_ms is not None else None
    )
    return RequestMetric(
        request_id=request_id,
        timestamp=time.time(),
        model=model,
        selected_inference_path=selected_inference_path,
        routing_reason=routing_reason,
        backend_used=backend_used,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_latency_ms=total_latency_ms,
        ttft_ms=ttft_ms,
        inter_token_latency_ms=inter_token_latency_ms,
        tokens_per_sec=tokens_per_sec,
        latency_target_ms=latency_target_ms,
        latency_target_met=latency_target_met,
        prefix_cache_hit=prefix_cache_hit,
        estimated_kv_cache_mb=float(features.get("estimated_kv_cache_mb") or 0.0),
        gpu_memory_allocated_mb=features.get("gpu_memory_allocated_mb"),
        gpu_memory_reserved_mb=features.get("gpu_memory_reserved_mb"),
    )
