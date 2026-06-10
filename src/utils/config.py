from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "device": "auto",
        "dtype": "auto",
    },
    "server": {"host": "0.0.0.0", "port": 8000},
    "router": {
        "prefix_cache_enabled": True,
        "speculative_enabled": False,
        "int4_gemv_enabled": True,
        "vllm_enabled": False,
        "reject_impossible_latency_targets": False,
        "long_context_threshold_tokens": 4096,
        "short_prompt_threshold_tokens": 512,
        "short_decode_threshold_tokens": 256,
        "tight_latency_target_ms": 1000,
    },
    "prefix_cache": {"prefix_chars": 1024, "max_entries": 1000},
    "speculative": {"draft_model_name": None},
    "vllm": {"server_url": None, "request_timeout_s": 600},
    "metrics": {"output_path": "runs/metrics.jsonl"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and merge it over defaults."""
    if path is None:
        return deepcopy(DEFAULT_CONFIG)

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return _deep_merge(DEFAULT_CONFIG, loaded)
