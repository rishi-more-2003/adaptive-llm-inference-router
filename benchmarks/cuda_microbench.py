from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def benchmark_matmul(
    hidden_size: int,
    output_size: int,
    batch_size: int,
    iterations: int,
    warmup: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    device = torch.device("cuda")
    activations = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weights = torch.randn(hidden_size, output_size, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = activations @ weights
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _ = activations @ weights
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iterations
    flops = 2 * batch_size * hidden_size * output_size
    tflops = flops / (avg_ms / 1000) / 1e12
    return {
        "batch_size": batch_size,
        "avg_latency_ms": avg_ms,
        "throughput_tflops": tflops,
        "shape": {
            "activations": [batch_size, hidden_size],
            "weights": [hidden_size, output_size],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this microbenchmark requires an NVIDIA GPU.")

    dtype = getattr(torch, args.dtype)
    results = [
        benchmark_matmul(
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            batch_size=batch_size,
            iterations=args.iterations,
            warmup=args.warmup,
            dtype=dtype,
        )
        for batch_size in args.batch_sizes
    ]
    return {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "dtype": str(dtype),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA GEMV/GEMM-shaped microbenchmark.")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--output-size", type=int, default=4096)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="runs/cuda_microbench.json")
    args = parser.parse_args()

    summary = run(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
