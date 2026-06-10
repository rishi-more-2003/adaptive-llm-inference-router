from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def save_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure()
    df["client_latency_ms"].dropna().plot(kind="hist", bins=20)
    plt.xlabel("Client latency (ms)")
    plt.title("Latency Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_distribution.png")
    plt.close()


def save_by_path(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.groupby("selected_path")
    latency = grouped["client_latency_ms"].agg(["median", lambda values: values.quantile(0.95)])
    latency.columns = ["p50_latency_ms", "p95_latency_ms"]
    latency.plot(kind="bar")
    plt.ylabel("Latency (ms)")
    plt.title("p50/p95 Latency by Inference Path")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_by_path.png")
    plt.close()

    grouped["tokens_per_sec"].mean().plot(kind="bar")
    plt.ylabel("Tokens/sec")
    plt.title("Tokens/sec by Inference Path")
    plt.tight_layout()
    plt.savefig(output_dir / "tokens_per_sec_by_path.png")
    plt.close()


def save_counts(df: pd.DataFrame, output_dir: Path) -> None:
    df["selected_path"].value_counts().plot(kind="bar")
    plt.ylabel("Requests")
    plt.title("Path Selection Counts")
    plt.tight_layout()
    plt.savefig(output_dir / "path_counts.png")
    plt.close()


def save_target_attainment(df: pd.DataFrame, output_dir: Path) -> None:
    if "latency_target_met" not in df or df["latency_target_met"].dropna().empty:
        return
    df.dropna(subset=["latency_target_met"]).groupby("selected_path")["latency_target_met"].mean().plot(
        kind="bar"
    )
    plt.ylabel("Attainment")
    plt.ylim(0, 1)
    plt.title("Latency-Target Attainment by Path")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_target_attainment_by_path.png")
    plt.close()


def save_prefix_hit_rate(df: pd.DataFrame, output_dir: Path) -> None:
    if "prefix_cache_hit" not in df:
        return
    hits = df["prefix_cache_hit"].fillna(False).astype(bool)
    cumulative = hits.expanding().mean()
    plt.figure()
    cumulative.plot()
    plt.xlabel("Request")
    plt.ylabel("Cumulative hit rate")
    plt.title("Prefix-Cache Hit Rate Over Time")
    plt.tight_layout()
    plt.savefig(output_dir / "prefix_cache_hit_rate.png")
    plt.close()


def save_kv_memory(df: pd.DataFrame, output_dir: Path) -> None:
    if "estimated_kv_cache_mb" not in df:
        return
    df.groupby("workload")["estimated_kv_cache_mb"].mean().plot(kind="bar")
    plt.ylabel("Estimated KV cache (MB)")
    plt.title("Estimated KV-Cache Memory by Workload")
    plt.tight_layout()
    plt.savefig(output_dir / "kv_cache_memory_by_workload.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("--input", default="runs/benchmark_results.jsonl")
    parser.add_argument("--output", default="runs/plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(input_path)
    if df.empty:
        raise ValueError(f"No benchmark rows found in {input_path}")

    save_histogram(df, output_dir)
    save_by_path(df, output_dir)
    save_counts(df, output_dir)
    save_target_attainment(df, output_dir)
    save_prefix_hit_rate(df, output_dir)
    save_kv_memory(df, output_dir)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
