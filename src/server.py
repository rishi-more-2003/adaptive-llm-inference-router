from __future__ import annotations

import argparse
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from src.metrics.collector import MetricsCollector, build_request_metric
from src.router.router import AdaptiveInferenceRouter
from src.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    OpenAIResponse,
    RouteInspectRequest,
    Usage,
    chat_messages_to_prompt,
)
from src.utils.config import load_config
from src.utils.logging import configure_logging


def create_app(config_path: str | None = None) -> FastAPI:
    config = load_config(config_path)
    app = FastAPI(title="Adaptive LLM Inference Router")
    app.state.config = config
    app.state.router = AdaptiveInferenceRouter(config)
    app.state.metrics = MetricsCollector(config.get("metrics", {}).get("output_path", "runs/metrics.jsonl"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": config.get("model", {}).get("name"),
            "vllm_available": app.state.router.vllm_backend.available,
            "vllm_error": app.state.router.vllm_backend.availability_error,
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        model_name = config.get("model", {}).get("name")
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": None,
                    "owned_by": "local",
                    "backend_capabilities": {
                        "hf_baseline": True,
                        "vllm": app.state.router.vllm_backend.available,
                        "int4_gemv_stub": True,
                        "prefix_cache_simulated": True,
                        "speculative_stub": True,
                        "long_context_stub": True,
                    },
                }
            ],
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return app.state.metrics.aggregate(app.state.router.prefix_cache.metrics())

    @app.get("/metrics/prometheus", response_class=PlainTextResponse)
    def prometheus_metrics() -> str:
        return app.state.metrics.prometheus_text(app.state.router.prefix_cache.metrics())

    @app.post("/v1/route")
    def inspect_route(request: RouteInspectRequest) -> dict[str, Any]:
        prompt = _route_prompt(request)
        decision = app.state.router.inspect_route(
            prompt=prompt,
            max_tokens=request.max_tokens,
            latency_target_ms=request.latency_target_ms,
            batch_size=request.batch_size,
            update_prefix_cache=request.update_prefix_cache,
        )
        return decision.to_dict()

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> OpenAIResponse:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not implemented in v1.")
        prompt = chat_messages_to_prompt(request.messages)
        routed = app.state.router.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            latency_target_ms=request.latency_target_ms,
            batch_size=request.batch_size,
        )
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        _record_metric(app, request_id, request.model, routed, request.latency_target_ms)
        result = routed.result

        return OpenAIResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result.text),
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.generated_tokens,
                total_tokens=result.prompt_tokens + result.generated_tokens,
            ),
            routing=routed.decision.to_dict(),
            metrics=_response_metrics(result),
        )

    @app.post("/v1/completions")
    def completions(request: CompletionRequest) -> OpenAIResponse:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not implemented in v1.")
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        prompt = prompts[0]
        batch_size = request.batch_size or len(prompts)
        routed = app.state.router.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            latency_target_ms=request.latency_target_ms,
            batch_size=batch_size,
        )
        request_id = f"cmpl-{uuid.uuid4().hex}"
        _record_metric(app, request_id, request.model, routed, request.latency_target_ms)
        result = routed.result

        return OpenAIResponse(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=[CompletionChoice(index=0, text=result.text)],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.generated_tokens,
                total_tokens=result.prompt_tokens + result.generated_tokens,
            ),
            routing=routed.decision.to_dict(),
            metrics=_response_metrics(result),
        )

    return app


def _route_prompt(request: RouteInspectRequest) -> str:
    if request.prompt is not None:
        return request.prompt
    if request.messages:
        return chat_messages_to_prompt(request.messages)
    raise HTTPException(status_code=400, detail="Route inspection requires either `prompt` or `messages`.")


def _response_metrics(result: Any) -> dict[str, Any]:
    return {
        "backend_name": result.backend_name,
        "total_latency_ms": result.total_latency_ms,
        "ttft_ms": result.ttft_ms,
        "tokens_per_sec": result.tokens_per_sec,
        "metadata": result.metadata,
    }


def _record_metric(
    app: FastAPI,
    request_id: str,
    model: str,
    routed: Any,
    latency_target_ms: int | None,
) -> None:
    decision = routed.decision
    result = routed.result
    metric = build_request_metric(
        request_id=request_id,
        model=model,
        selected_inference_path=decision.path.value,
        routing_reason=decision.reason,
        backend_used=result.backend_name,
        prompt_tokens=result.prompt_tokens,
        generated_tokens=result.generated_tokens,
        total_latency_ms=result.total_latency_ms,
        ttft_ms=result.ttft_ms,
        tokens_per_sec=result.tokens_per_sec,
        latency_target_ms=latency_target_ms,
        prefix_cache_hit=bool(decision.features.get("prefix_cache_hit")),
        features=decision.features,
    )
    app.state.metrics.record(metric)


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Adaptive LLM Inference Router server.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    server_cfg = config.get("server", {})
    uvicorn.run(
        create_app(args.config),
        host=str(server_cfg.get("host", "0.0.0.0")),
        port=int(server_cfg.get("port", 8000)),
    )


if __name__ == "__main__":
    main()
