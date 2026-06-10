from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadRequest:
    workload: str
    prompt: str
    expected_path: str


SHARED_SYSTEM_PREFIX = " ".join(
    [
        "You are an ML infrastructure assistant. Answer with practical systems detail.",
        "Focus on LLM serving, latency, throughput, memory bandwidth, KV cache pressure,",
        "prefix reuse, speculative decoding, and backend selection.",
    ]
    * 12
)

CODE_PROMPTS = [
    "Write code for a Python function that estimates p95 latency from a list of floats.",
    "Write code for a Python helper that computes tokens per second from token and latency counts.",
    "Write code for a FastAPI health endpoint that returns model and backend status.",
    "Write code for a small JSONL writer for benchmark metrics.",
]


def batch1_short_decode() -> WorkloadRequest:
    prompts = [
        "Explain why batch-1 LLM decode is often memory-bandwidth-bound.",
        "Summarize CUDA memory coalescing in two paragraphs.",
        "Why can GEMV win for single-token decode?",
    ]
    return WorkloadRequest("batch1_short_decode", random.choice(prompts), "int4_gemv")


def shared_prefix_chat() -> WorkloadRequest:
    questions = [
        "What metrics should an inference router log?",
        "How does prefix caching reduce prefill cost?",
        "Why does workload shape change the fastest backend?",
    ]
    return WorkloadRequest(
        "shared_prefix_chat",
        f"{SHARED_SYSTEM_PREFIX}\n\nUser question: {random.choice(questions)}",
        "prefix_cache",
    )


def long_context_qa() -> WorkloadRequest:
    context = " ".join(
        f"Document chunk {idx}: LLM serving systems must balance KV cache memory and latency."
        for idx in range(450)
    )
    return WorkloadRequest(
        "long_context_qa",
        f"Context:\n{context}\n\nQuestion: What bottleneck dominates this request?",
        "long_context",
    )


def code_completion() -> WorkloadRequest:
    return WorkloadRequest(
        "code_completion",
        random.choice(CODE_PROMPTS),
        "baseline",
    )


def tight_latency_target() -> WorkloadRequest:
    return WorkloadRequest(
        "tight_latency_target",
        "Give one sentence about speculative decoding.",
        "speculative_if_enabled",
    )


GENERATORS = {
    "batch1_short_decode": batch1_short_decode,
    "shared_prefix_chat": shared_prefix_chat,
    "long_context_qa": long_context_qa,
    "code_completion": code_completion,
    "tight_latency_target": tight_latency_target,
}


def generate_workload(name: str, num_requests: int) -> list[WorkloadRequest]:
    if name == "mixed":
        generators = list(GENERATORS.values())
        requests: list[WorkloadRequest] = []
        while len(requests) < num_requests:
            for generator in generators:
                if len(requests) >= num_requests:
                    break
                requests.append(generator())
        return requests
    if name == "code_completion":
        return [
            WorkloadRequest("code_completion", CODE_PROMPTS[index % len(CODE_PROMPTS)], "baseline")
            for index in range(num_requests)
        ]
    if name not in GENERATORS:
        valid = ", ".join(["mixed", *GENERATORS])
        raise ValueError(f"Unknown workload '{name}'. Valid workloads: {valid}")
    return [GENERATORS[name]() for _ in range(num_requests)]
