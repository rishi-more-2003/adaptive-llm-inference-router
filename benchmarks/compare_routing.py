from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

try:
    from benchmarks.run_benchmark import expected_path_matches
    from benchmarks.workloads import generate_workload
except ModuleNotFoundError:
    from run_benchmark import expected_path_matches
    from workloads import generate_workload


STATIC_POLICIES = ("baseline", "vllm", "int4_gemv")


def inspect_route(
    server: str,
    prompt: str,
    max_tokens: int,
    latency_target_ms: int,
    update_prefix_cache: bool,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "latency_target_ms": latency_target_ms,
        "batch_size": 1,
        "update_prefix_cache": update_prefix_cache,
    }
    response = httpx.post(f"{server.rstrip('/')}/v1/route", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def compare(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for request in generate_workload(args.workload, args.num_requests):
        decision = inspect_route(
            args.server,
            request.prompt,
            args.max_tokens,
            args.latency_target_ms,
            update_prefix_cache=request.workload == "shared_prefix_chat",
        )
        adaptive_path = decision["path"]
        row = {
            "workload": request.workload,
            "expected_path": request.expected_path,
            "adaptive_path": adaptive_path,
            "adaptive_match": expected_path_matches(request.expected_path, adaptive_path),
            "static_matches": {
                policy: expected_path_matches(request.expected_path, policy) for policy in STATIC_POLICIES
            },
            "routing_reason": decision["reason"],
            "latency_prediction": decision.get("features", {}).get("latency_prediction"),
        }
        rows.append(row)

    adaptive_evaluable = [row for row in rows if row["adaptive_match"] is not None]
    static_evaluable = {
        policy: [row for row in rows if row["static_matches"][policy] is not None]
        for policy in STATIC_POLICIES
    }

    summary = {
        "requests": len(rows),
        "workload_counts": dict(Counter(row["workload"] for row in rows)),
        "adaptive_path_counts": dict(Counter(row["adaptive_path"] for row in rows)),
        "adaptive_match_rate": _match_rate(row["adaptive_match"] for row in adaptive_evaluable),
        "static_match_rates": {
            policy: _match_rate(row["static_matches"][policy] for row in static_evaluable[policy])
            for policy in STATIC_POLICIES
        },
        "rows": rows,
    }
    return summary


def _match_rate(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare adaptive route selection against static policies.")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--workload", default="mixed")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--latency-target-ms", type=int, default=1000)
    parser.add_argument("--output", default="runs/routing_comparison.json")
    args = parser.parse_args()

    summary = compare(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
