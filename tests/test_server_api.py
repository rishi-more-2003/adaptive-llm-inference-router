from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.backends.base import GenerationResult
from src.metrics.collector import MetricsCollector
from src.router.policy import InferencePath, RoutingDecision
from src.router.router import RoutedGeneration
from src.server import create_app


class FakePrefixCache:
    def metrics(self) -> dict[str, float | int]:
        return {
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 1,
            "hit_rate": 0.0,
            "estimated_tokens_saved": 0,
            "entries": 1,
        }


class FakeVllmBackend:
    available = False
    availability_error = "not available in tests"


class FakeRouter:
    def __init__(self) -> None:
        self.vllm_backend = FakeVllmBackend()
        self.prefix_cache = FakePrefixCache()
        self.last_batch_size: int | None = None

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        latency_target_ms: int | None = None,
        batch_size: int = 1,
    ) -> RoutedGeneration:
        self.last_batch_size = batch_size
        features: dict[str, Any] = {
            "prompt_chars": len(prompt),
            "estimated_prompt_tokens": 4,
            "max_new_tokens": max_tokens,
            "batch_size": batch_size,
            "latency_target_ms": latency_target_ms,
            "prefix_cache_hit": False,
            "estimated_kv_cache_mb": 1.0,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
        }
        return RoutedGeneration(
            result=GenerationResult(
                text="fake response",
                prompt_tokens=4,
                generated_tokens=2,
                total_latency_ms=12.5,
                ttft_ms=6.25,
                tokens_per_sec=160.0,
                backend_name="fake_backend",
                metadata={"test": True},
            ),
            decision=RoutingDecision(
                path=InferencePath.BASELINE,
                reason="fake route for API test",
                features=features,
            ),
            prefix_cache_metrics=self.prefix_cache.metrics(),
        )

    def inspect_route(
        self,
        prompt: str,
        max_tokens: int,
        latency_target_ms: int | None = None,
        batch_size: int = 1,
        update_prefix_cache: bool = False,
    ) -> RoutingDecision:
        self.last_batch_size = batch_size
        return RoutingDecision(
            path=InferencePath.INT4_GEMV,
            reason="fake inspect route",
            features={
                "prompt_chars": len(prompt),
                "estimated_prompt_tokens": 4,
                "max_new_tokens": max_tokens,
                "batch_size": batch_size,
                "latency_target_ms": latency_target_ms,
                "prefix_cache_hit": update_prefix_cache,
                "estimated_kv_cache_mb": 1.0,
            },
        )


def test_health_endpoint_reports_router_state(tmp_path: Path):
    app = create_test_app(tmp_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["vllm_available"] is False
    assert response.json()["vllm_error"] == "not available in tests"


def test_models_endpoint_reports_configured_model_and_capabilities(tmp_path: Path):
    app = create_test_app(tmp_path)
    client = TestClient(app)

    response = client.get("/v1/models")

    data = response.json()
    assert response.status_code == 200
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert data["data"][0]["backend_capabilities"]["hf_baseline"] is True
    assert data["data"][0]["backend_capabilities"]["vllm"] is False


def test_chat_completion_response_shape_and_metrics(tmp_path: Path):
    app = create_test_app(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "batch_size": 2,
        },
    )

    data = response.json()
    assert response.status_code == 200
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "fake response"
    assert data["usage"]["total_tokens"] == 6
    assert data["routing"]["features"]["batch_size"] == 2
    assert app.state.router.last_batch_size == 2

    metrics = client.get("/metrics").json()
    assert metrics["requests"] == 1
    assert metrics["path_counts"] == {"baseline": 1}

    prometheus = client.get("/metrics/prometheus")
    assert prometheus.status_code == 200
    assert "adaptive_router_requests_total 1" in prometheus.text
    assert 'adaptive_router_path_requests_total{path="baseline"} 1' in prometheus.text


def test_route_inspection_endpoint_does_not_generate(tmp_path: Path):
    app = create_test_app(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/v1/route",
        json={
            "prompt": "Explain batch-1 decode.",
            "max_tokens": 16,
            "batch_size": 3,
            "latency_target_ms": 500,
        },
    )

    data = response.json()
    assert response.status_code == 200
    assert data["path"] == "int4_gemv"
    assert data["features"]["batch_size"] == 3
    assert app.state.router.last_batch_size == 3
    assert client.get("/metrics").json()["requests"] == 0


def test_route_inspection_requires_prompt_or_messages(tmp_path: Path):
    app = create_test_app(tmp_path)
    client = TestClient(app)

    response = client.post("/v1/route", json={"max_tokens": 16})

    assert response.status_code == 400
    assert "requires either `prompt` or `messages`" in response.json()["detail"]


def test_completion_list_prompt_sets_batch_size(tmp_path: Path):
    app = create_test_app(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": ["first prompt", "second prompt", "third prompt"],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 200
    assert app.state.router.last_batch_size == 3


def test_streaming_request_is_rejected_before_generation(tmp_path: Path):
    app = create_test_app(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert "Streaming is not implemented" in response.json()["detail"]
    assert app.state.router.last_batch_size is None


def create_test_app(tmp_path: Path):
    app = create_app(None)
    app.state.router = FakeRouter()
    app.state.metrics = MetricsCollector(tmp_path / "metrics.jsonl")
    return app
