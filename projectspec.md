# Project Spec: Adaptive LLM Inference Router

## Repository Name

`adaptive-llm-inference-router`

## One-Line Goal

Build a local LLM serving engine that routes each request to the best inference path based on workload features such as batch size, prompt length, prefix reuse, KV-cache pressure, decode length, and latency target.

## Project Thesis

The fastest LLM inference path is workload-dependent.

Batch-1 short decode can be memory-bandwidth-bound and benefit from an INT4 GEMV-style path. Batch-2+ may favor GEMM or batched serving. Repeated prompts benefit from prefix-cache reuse. Long-context requests are often KV-cache-pressure-bound. Speculative decoding can help when draft-token acceptance is high.

This project builds a router that measures these request features and selects the best inference path automatically.

## Important Non-Goal

This project is **not** a multi-SLO request scheduler.

It does not primarily optimize:

* global request ordering
* multi-tenant priority assignment
* queue placement
* simulated annealing over batches
* global SLO-attainment scheduling

Those ideas overlap too closely with existing work such as “SLO-Aware Scheduling for Large Language Model Inferences” by Cheng et al.

Instead, this project focuses on **per-request inference-path selection**:

```text
Given this request and current runtime state, which execution path should handle it?
```

The router chooses between:

* baseline HuggingFace generation
* optional vLLM backend
* INT4 GEMV decode path stub
* prefix-cache reuse path
* speculative decoding path stub
* long-context/KV-aware path

## Target Audience

This repo should look credible to ML infrastructure, inference, CUDA, and LLM serving teams.

The project should communicate:

* inference profiling ability
* systems thinking
* knowledge of decode bottlenecks
* workload-aware optimization
* clean benchmarking
* extensible backend design
* practical ML infra engineering

## Main Project Name

**Adaptive LLM Inference Router**

Subtitle:

**Workload-aware routing across decode, cache, speculative, and long-context inference paths.**

## README Positioning

Use this language:

```text
My previous project optimized batch-1 decode with a fused INT4 dequant + GEMV kernel. But batch-2 changed the optimal path: GEMM weight reuse started winning again.

That observation motivates this project.

LLM inference optimization is not one-size-fits-all. The best path depends on batch size, prompt length, prefix reuse, KV-cache pressure, and latency target. This repo builds a local serving router that classifies each request and selects the most appropriate inference path.
```

## Core Features

### 1. OpenAI-Compatible Local Server

Implement a FastAPI server exposing:

```text
POST /v1/chat/completions
POST /v1/completions
GET /health
GET /metrics
```

Support a minimal OpenAI-like request schema:

```json
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "messages": [
    {"role": "user", "content": "Explain why batch-1 LLM decode is memory-bandwidth-bound."}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "stream": false,
  "latency_target_ms": 1000
}
```

Return an OpenAI-like response.

Streaming is optional for v1. Prioritize a correct non-streaming path first.

### 2. Backend Abstraction

Create a clean backend interface.

File:

```text
src/backends/base.py
```

Interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    total_latency_ms: float
    ttft_ms: Optional[float]
    tokens_per_sec: float
    backend_name: str
    metadata: Dict[str, Any]


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        pass
```

Implement the following backends.

### 3. HuggingFace Baseline Backend

File:

```text
src/backends/hf_backend.py
```

Use HuggingFace Transformers as the default reliable backend.

Requirements:

* Default model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* CUDA if available
* CPU fallback
* automatic dtype selection
* prompt token counting
* generated token counting
* total latency measurement
* tokens/sec measurement
* approximate TTFT if true token-level streaming is not implemented

This backend should always work as the baseline.

### 4. vLLM Backend

File:

```text
src/backends/vllm_backend.py
```

Make vLLM optional.

Requirements:

* If vLLM is installed, use it.
* If vLLM is not installed, fail gracefully.
* The repo must still run without vLLM.
* Add clear setup instructions in README.
* Keep this backend modular.

Do not place `vllm` in base `requirements.txt`.

Add optional install section:

```bash
pip install vllm
```

### 5. INT4 GEMV Backend Stub

File:

```text
src/backends/int4_gemv_backend.py
```

This is an integration point for a future fused INT4 dequant + GEMV kernel.

For v1:

* Wrap the HuggingFace backend.
* Log that the INT4 path was selected.
* Add clean TODOs for CUDA/Triton integration.
* Expose a future method like:

```python
def decode_one_token_int4(self, hidden_state, quantized_weights, scales):
    """
    Future extension point for fused INT4 dequant + GEMV decode.
    """
    raise NotImplementedError
```

Important: do not pretend this backend has a real custom kernel yet. Label it clearly as a stub.

### 6. Prefix Cache Simulator

File:

```text
src/cache/prefix_cache.py
```

Implement a lightweight prefix-cache simulator.

This does not need to manipulate real KV-cache blocks in v1. It should simulate prefix reuse and expose metrics.

Features:

* normalize prompt text
* hash prefixes
* track hits and misses
* estimate saved prompt tokens
* expose hit rate
* configurable prefix length
* configurable max cache entries

Prefix match logic:

```text
normalize whitespace
take first N characters or first N estimated tokens
hash the prefix
check whether prefix exists
```

Metrics:

```json
{
  "prefix_cache_hits": 12,
  "prefix_cache_misses": 38,
  "hit_rate": 0.24,
  "estimated_tokens_saved": 8192
}
```

### 7. Speculative Decoding Backend Stub

File:

```text
src/backends/speculative_backend.py
```

For v1:

* Implement as a wrapper around the HuggingFace backend.
* Log that speculative decoding was selected.
* Do not claim real speculative decoding speedups.
* Add clean extension points for a draft model.

Future interface:

```python
class SpeculativeBackend(InferenceBackend):
    def generate_with_draft_model(
        self,
        prompt: str,
        draft_model_name: str,
        target_model_name: str,
        max_tokens: int,
    ) -> GenerationResult:
        raise NotImplementedError
```

### 8. Long-Context Backend Stub

File:

```text
src/backends/long_context_backend.py
```

For v1:

* Wrap the HuggingFace backend.
* Log that long-context mode was selected.
* Track estimated KV-cache memory.
* Add TODOs for paged KV cache, FP8 KV cache, or vLLM integration.

### 9. Request Feature Extractor

File:

```text
src/router/features.py
```

Extract request-level features:

```python
@dataclass
class RequestFeatures:
    prompt_chars: int
    estimated_prompt_tokens: int
    max_new_tokens: int
    batch_size: int
    latency_target_ms: Optional[int]
    has_shared_prefix: bool
    prefix_cache_hit: bool
    estimated_kv_cache_mb: float
    request_type: str
    cuda_available: bool
    gpu_memory_allocated_mb: Optional[float]
    gpu_memory_reserved_mb: Optional[float]
```

Request type classifier can be rule-based:

```text
contains code fences or "write code" → code
prompt length > long_context_threshold → long_context
contains "context:" or "documents:" → rag
otherwise → chat
```

### 10. Adaptive Router

File:

```text
src/router/policy.py
```

Represent inference paths as an enum:

```python
from enum import Enum


class InferencePath(Enum):
    BASELINE = "baseline"
    VLLM = "vllm"
    INT4_GEMV = "int4_gemv"
    PREFIX_CACHE = "prefix_cache"
    SPECULATIVE = "speculative"
    LONG_CONTEXT = "long_context"
```

Policy decision object:

```python
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RoutingDecision:
    path: InferencePath
    reason: str
    features: Dict[str, Any]
```

Initial routing rules:

```text
if prefix_cache_hit and prefix_cache_enabled:
    choose PREFIX_CACHE

else if estimated_prompt_tokens > long_context_threshold_tokens:
    choose LONG_CONTEXT

else if batch_size == 1 and estimated_prompt_tokens < short_prompt_threshold_tokens and max_new_tokens <= short_decode_threshold_tokens and int4_gemv_enabled:
    choose INT4_GEMV

else if latency_target_ms is tight and speculative_enabled:
    choose SPECULATIVE

else if vllm_enabled and vllm_available:
    choose VLLM

else:
    choose BASELINE
```

Default thresholds:

```yaml
router:
  short_prompt_threshold_tokens: 512
  short_decode_threshold_tokens: 256
  long_context_threshold_tokens: 4096
  tight_latency_target_ms: 1000
  prefix_cache_enabled: true
  int4_gemv_enabled: true
  speculative_enabled: false
  vllm_enabled: false
```

Every routing decision should return an explanation:

```json
{
  "path": "int4_gemv",
  "reason": "batch_size=1, prompt_tokens=184, max_new_tokens=128; expected memory-bandwidth-bound decode path",
  "features": {
    "estimated_prompt_tokens": 184,
    "max_new_tokens": 128,
    "batch_size": 1,
    "latency_target_ms": 1000,
    "prefix_cache_hit": false,
    "estimated_kv_cache_mb": 32.1
  }
}
```

### 11. Metrics System

File:

```text
src/metrics/collector.py
```

Track per request:

* request id
* timestamp
* model
* selected inference path
* routing reason
* backend used
* prompt tokens
* generated tokens
* total latency ms
* TTFT ms, if available
* inter-token latency ms
* tokens/sec
* latency target ms
* latency target met
* prefix cache hit
* estimated KV-cache memory
* GPU memory allocated
* GPU memory reserved

Save metrics to:

```text
runs/metrics.jsonl
```

Expose aggregate metrics:

```text
GET /metrics
```

Example response:

```json
{
  "requests": 100,
  "p50_latency_ms": 420,
  "p95_latency_ms": 980,
  "avg_tokens_per_sec": 37.2,
  "latency_target_attainment": 0.91,
  "path_counts": {
    "baseline": 40,
    "int4_gemv": 30,
    "prefix_cache": 20,
    "speculative": 5,
    "long_context": 5
  },
  "prefix_cache_hit_rate": 0.27
}
```

### 12. Benchmark Harness

File:

```text
benchmarks/run_benchmark.py
```

Generate synthetic workloads and call the local server.

Command:

```bash
python benchmarks/run_benchmark.py \
  --server http://localhost:8000 \
  --workload mixed \
  --num-requests 100 \
  --concurrency 4 \
  --max-tokens 128 \
  --latency-target-ms 1000
```

Workloads:

#### A. Batch-1 Short Decode

Small prompts, short outputs.

Examples:

```text
Explain why batch-1 LLM decode is often memory-bandwidth-bound.
```

Expected path:

```text
INT4_GEMV
```

#### B. Shared Prefix Chat

Repeated system prompt with different user questions.

Expected path after warmup:

```text
PREFIX_CACHE
```

#### C. Long-Context QA

Large context with a question at the end.

Expected path:

```text
LONG_CONTEXT
```

#### D. Code Completion

Code-heavy prompts.

Expected path:

```text
BASELINE or VLLM
```

#### E. Tight Latency Target

Short prompt with a tight latency target.

Expected path if enabled:

```text
SPECULATIVE
```

#### F. Mixed Workload

Randomly combine all above workloads.

Outputs:

* console summary table
* raw results: `runs/benchmark_results.jsonl`
* summary: `runs/benchmark_summary.json`

### 13. Visualization Script

File:

```text
benchmarks/plot_results.py
```

Generate plots:

* latency distribution
* p50/p95 latency by inference path
* tokens/sec by inference path
* latency-target attainment by path
* prefix-cache hit rate over time
* path selection counts
* estimated KV-cache memory by workload

Save plots to:

```text
runs/plots/
```

Use matplotlib only.

### 14. Config System

File:

```text
configs/default.yaml
```

Example:

```yaml
model:
  name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  device: auto
  dtype: auto

server:
  host: 0.0.0.0
  port: 8000

router:
  prefix_cache_enabled: true
  speculative_enabled: false
  int4_gemv_enabled: true
  vllm_enabled: false
  long_context_threshold_tokens: 4096
  short_prompt_threshold_tokens: 512
  short_decode_threshold_tokens: 256
  tight_latency_target_ms: 1000

prefix_cache:
  prefix_chars: 1024
  max_entries: 1000

metrics:
  output_path: runs/metrics.jsonl
```

### 15. CLI

Create:

```text
src/cli.py
```

Support:

```bash
python -m src.cli inspect-route \
  --prompt "Explain CUDA memory coalescing." \
  --max-tokens 128 \
  --latency-target-ms 1000
```

Expected output:

```json
{
  "path": "int4_gemv",
  "reason": "batch_size=1, prompt_tokens=42, max_new_tokens=128; expected memory-bandwidth-bound decode path",
  "features": {
    "estimated_prompt_tokens": 42,
    "max_new_tokens": 128,
    "batch_size": 1,
    "latency_target_ms": 1000,
    "prefix_cache_hit": false
  }
}
```

Also support:

```bash
python -m src.server --config configs/default.yaml
```

### 16. Suggested File Tree

```text
adaptive-llm-inference-router/
  README.md
  projectspec.md
  pyproject.toml
  requirements.txt
  configs/
    default.yaml
  src/
    __init__.py
    server.py
    cli.py
    schemas.py
    backends/
      __init__.py
      base.py
      hf_backend.py
      vllm_backend.py
      int4_gemv_backend.py
      speculative_backend.py
      long_context_backend.py
    router/
      __init__.py
      features.py
      policy.py
      router.py
    cache/
      __init__.py
      prefix_cache.py
    metrics/
      __init__.py
      collector.py
    utils/
      __init__.py
      logging.py
      tokens.py
      gpu.py
      config.py
  benchmarks/
    run_benchmark.py
    plot_results.py
    workloads.py
  tests/
    test_router_policy.py
    test_prefix_cache.py
    test_feature_extractor.py
  runs/
    .gitkeep
```

### 17. Requirements

Base `requirements.txt`:

```text
fastapi
uvicorn
pydantic
pyyaml
transformers
torch
accelerate
numpy
pandas
matplotlib
httpx
tqdm
pytest
```

Optional:

```text
vllm
```

Do not require vLLM for the base repo.

### 18. README Structure

Write a strong README with this structure:

```text
# Adaptive LLM Inference Router

A local LLM serving engine that routes requests across inference paths based on workload features: batch size, prompt length, prefix reuse, KV-cache pressure, decode length, and latency target.

## Why this exists

My previous project optimized batch-1 decode with a fused INT4 dequant + GEMV kernel. But batch-2 changed the optimal path: GEMM weight reuse started winning again.

This repo explores the systems-level consequence: LLM inference optimization is workload-dependent.

## What this is not

This is not a multi-SLO request scheduler. It does not optimize global request ordering or queue priority assignment.

Instead, it performs per-request inference-path routing.

## Features

- OpenAI-compatible local API
- Workload-aware inference router
- HuggingFace baseline backend
- Optional vLLM backend
- INT4 GEMV integration stub
- Prefix-cache simulator
- Speculative decoding stub
- Long-context path stub
- Benchmark harness
- TTFT, p95 latency, tokens/sec, path-selection metrics
- Visualization scripts

## Architecture

Include ASCII diagram.

## Quickstart

## Run Server

## Inspect Route

## Run Benchmark

## Example Routing Decision

## Results

Placeholder benchmark table.

## Related Work

Mention:
- vLLM
- PagedAttention
- speculative decoding
- prefix caching
- SLO-Aware Scheduling for Large Language Model Inferences

Clearly distinguish this project:
- related work schedules requests globally
- this repo routes requests across inference execution paths

## Roadmap

- real vLLM backend
- real prefix-cache integration
- real speculative decoding
- real INT4 GEMV CUDA/Triton backend
- continuous batching
- latency prediction
- admission control
- Prometheus metrics
- Dockerfile
```

Architecture diagram:

```text
Client
  |
  v
FastAPI OpenAI-Compatible Server
  |
  v
Request Feature Extractor
  |
  v
Adaptive Inference Router
  |
  |------------|-------------|--------------|----------------|----------------|
  v            v             v              v                v
HF Baseline   vLLM       INT4 GEMV Stub   Prefix Cache    Long Context
  |
  v
Metrics Collector -> JSONL -> Benchmark Summary -> Plots
```

### 19. Related Work Section Wording

Use this in README:

```text
## Related Work

This project is adjacent to LLM serving systems such as vLLM, PagedAttention, prefix caching, speculative decoding, and SLO-aware inference scheduling.

A closely related paper is “SLO-Aware Scheduling for Large Language Model Inferences” by Cheng et al. That work focuses on multi-SLO request scheduling: deciding priority order and batching strategy to improve global SLO attainment.

This repo intentionally tackles a different layer of the stack. It does not optimize global request ordering. Instead, it performs per-request inference-path routing: deciding whether a request should use baseline decode, a batch-1 INT4 GEMV path, prefix-cache reuse, speculative decoding, or a long-context/KV-aware path.
```

### 20. Example Commands

Install:

```bash
pip install -r requirements.txt
```

Run server:

```bash
python -m src.server --config configs/default.yaml
```

Health check:

```bash
curl http://localhost:8000/health
```

Chat completion:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "user", "content": "Explain why batch-1 LLM decode is memory-bandwidth-bound."}
    ],
    "max_tokens": 128,
    "temperature": 0.7,
    "latency_target_ms": 1000
  }'
```

Inspect route:

```bash
python -m src.cli inspect-route \
  --prompt "Explain CUDA memory coalescing." \
  --max-tokens 128 \
  --latency-target-ms 1000
```

Run benchmark:

```bash
python benchmarks/run_benchmark.py \
  --workload mixed \
  --num-requests 50 \
  --concurrency 4 \
  --max-tokens 128 \
  --latency-target-ms 1000
```

Plot results:

```bash
python benchmarks/plot_results.py \
  --input runs/benchmark_results.jsonl \
  --output runs/plots
```

### 21. Unit Tests

Add tests for:

```text
tests/test_router_policy.py
tests/test_prefix_cache.py
tests/test_feature_extractor.py
```

Minimum test cases:

#### Router Policy

* short batch-1 request selects INT4_GEMV
* repeated prefix selects PREFIX_CACHE
* long prompt selects LONG_CONTEXT
* tight latency target selects SPECULATIVE only when enabled
* fallback selects BASELINE

#### Prefix Cache

* first request is miss
* repeated prefix is hit
* whitespace normalization works
* max cache size is respected

#### Feature Extractor

* prompt token estimate is positive
* code prompt classified as code
* long context classified as long_context
* RAG-style prompt classified as rag

### 22. Implementation Order

Build in this order:

#### Phase 1: Skeleton

* repo structure
* config loader
* schemas
* FastAPI server
* `/health`
* HuggingFace backend

#### Phase 2: Router

* request feature extractor
* inference path enum
* rule-based router
* inspect-route CLI
* router tests

#### Phase 3: Metrics

* per-request metrics
* JSONL logging
* `/metrics`
* aggregate stats

#### Phase 4: Prefix Cache

* prefix cache simulator
* hit/miss metrics
* connect prefix hits to router

#### Phase 5: Backend Stubs

* INT4 GEMV backend stub
* speculative backend stub
* long-context backend stub
* optional vLLM backend

#### Phase 6: Benchmarking

* workload generator
* benchmark runner
* concurrent request support
* benchmark summary

#### Phase 7: Visualization and README

* plotting script
* architecture diagram
* related work section
* result placeholders
* roadmap

### 23. Expected v1 Quality

The final v1 should run with:

```bash
pip install -r requirements.txt
python -m src.server --config configs/default.yaml
```

Then:

```bash
python benchmarks/run_benchmark.py --workload mixed --num-requests 20
```

should produce:

```text
runs/benchmark_results.jsonl
runs/benchmark_summary.json
```

and:

```bash
python benchmarks/plot_results.py
```

should produce plots under:

```text
runs/plots/
```

### 24. Honesty Requirements

Do not fake benchmark results.

Do not claim that INT4 GEMV, speculative decoding, or long-context optimization is fully implemented if those paths are stubs.

Use honest labels:

```text
INT4_GEMV_STUB
SPECULATIVE_STUB
LONG_CONTEXT_STUB
PREFIX_CACHE_SIMULATED
```

The v1 contribution is still meaningful because it provides:

* real serving API
* real routing logic
* real feature extraction
* real metrics
* real benchmark harness
* real prefix-cache simulation
* clean extension points for CUDA and vLLM integration

### 25. Stretch Goals

After v1:

1. Integrate real vLLM backend.
2. Pull real prefix-cache metrics from vLLM.
3. Add draft-model speculative decoding.
4. Integrate fused INT4 dequant + GEMV kernel.
5. Add continuous batching.
6. Add latency prediction.
7. Add admission control for impossible latency targets.
8. Add Prometheus endpoint.
9. Add Dockerfile.
10. Add GitHub Actions.
11. Add CUDA/Triton microbenchmarks.
12. Compare routing decisions against static backend selection.

### 26. Final Design Principle

The project should make one clear argument:

```text
LLM inference optimization is workload-dependent. The correct execution path changes with batch size, prompt length, prefix reuse, KV-cache pressure, and decode length. This repo builds a measurable router that makes that choice explicit.
```

Build everything around that.
