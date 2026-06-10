from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

try:
    from benchmarks.workloads import WorkloadRequest, generate_workload
except ModuleNotFoundError:
    from workloads import WorkloadRequest, generate_workload


async def send_request(
    client: httpx.AsyncClient,
    server: str,
    request: WorkloadRequest,
    max_tokens: int,
    latency_target_ms: int | None,
    batch_size: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "latency_target_ms": latency_target_ms,
        "batch_size": batch_size,
    }
    async with semaphore:
        started = time.perf_counter()
        response = await client.post(f"{server.rstrip('/')}/v1/chat/completions", json=payload)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data = response.json()
        routing = data.get("routing", {})
        metrics = data.get("metrics", {})
        return {
            "workload": request.workload,
            "expected_path": request.expected_path,
            "selected_path": routing.get("path"),
            "expected_path_matched": expected_path_matches(request.expected_path, routing.get("path")),
            "routing_reason": routing.get("reason"),
            "backend_name": metrics.get("backend_name"),
            "client_latency_ms": elapsed_ms,
            "server_latency_ms": metrics.get("total_latency_ms"),
            "tokens_per_sec": metrics.get("tokens_per_sec"),
            "latency_target_ms": latency_target_ms,
            "latency_target_met": elapsed_ms <= latency_target_ms if latency_target_ms else None,
            "prefix_cache_hit": routing.get("features", {}).get("prefix_cache_hit"),
            "estimated_kv_cache_mb": routing.get("features", {}).get("estimated_kv_cache_mb"),
        }


async def run_async(args: argparse.Namespace) -> list[dict[str, Any]]:
    requests = generate_workload(args.workload, args.num_requests)
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            send_request(
                client,
                args.server,
                request,
                args.max_tokens,
                args.latency_target_ms,
                args.batch_size,
                semaphore,
            )
            for request in requests
        ]
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="benchmark"):
            results.append(await task)
        return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(float(row["client_latency_ms"]) for row in results)
    tokens_per_sec = [
        float(row["tokens_per_sec"]) for row in results if row.get("tokens_per_sec") is not None
    ]
    path_counts = Counter(row.get("selected_path") for row in results)
    workloads = Counter(row.get("workload") for row in results)
    target_rows = [row for row in results if row.get("latency_target_met") is not None]
    matched_rows = [row for row in results if row.get("expected_path_matched") is not None]

    return {
        "requests": len(results),
        "p50_client_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_client_latency_ms": percentile(latencies, 0.95) if latencies else None,
        "avg_tokens_per_sec": statistics.mean(tokens_per_sec) if tokens_per_sec else None,
        "latency_target_attainment": (
            sum(1 for row in target_rows if row["latency_target_met"]) / len(target_rows)
            if target_rows
            else None
        ),
        "path_counts": dict(path_counts),
        "workload_counts": dict(workloads),
        "expected_path_match_rate": (
            sum(1 for row in matched_rows if row["expected_path_matched"]) / len(matched_rows)
            if matched_rows
            else None
        ),
    }


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    index = min(len(values) - 1, max(0, round((len(values) - 1) * pct)))
    return values[index]


def expected_path_matches(expected_path: str, selected_path: str | None) -> bool | None:
    if expected_path == "speculative_if_enabled":
        return None
    if selected_path is None:
        return False
    if expected_path == "baseline":
        return selected_path in {"baseline", "vllm"}
    return selected_path == expected_path


def write_outputs(results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "benchmark_results.jsonl"
    summary_path = output_dir / "benchmark_summary.json"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")

    summary = summarize(results)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic router benchmark workloads.")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--workload", default="mixed")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--latency-target-ms", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()

    results = asyncio.run(run_async(args))
    summary = write_outputs(results, Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
