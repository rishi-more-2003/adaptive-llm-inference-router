from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from typing import Any

import httpx


def check_python() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "fastapi", "uvicorn", "httpx", "vllm"):
        try:
            packages[name] = {
                "installed": importlib.util.find_spec(name) is not None,
                "version": importlib.metadata.version(name),
            }
        except Exception as exc:
            packages[name] = {"installed": False, "version": None, "error": str(exc)}
    return {"platform": platform.platform(), "packages": packages}


def check_torch_cuda() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    result: dict[str, Any] = {
        "ok": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        result.update(
            {
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
                "memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 3),
                "memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 3),
            }
        )
    return result


def check_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def check_http(url: str, path: str) -> dict[str, Any]:
    full_url = f"{url.rstrip('/')}{path}"
    try:
        response = httpx.get(full_url, timeout=10)
        body: Any
        try:
            body = response.json()
        except Exception:
            body = response.text
        return {"ok": response.is_success, "status_code": response.status_code, "body": body}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local Adaptive LLM Inference Router environment.")
    parser.add_argument("--router-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()

    report = {
        "python": check_python(),
        "torch_cuda": check_torch_cuda(),
        "nvidia_smi": check_command(["nvidia-smi"]),
        "router_health": check_http(args.router_url, "/health"),
        "router_models": check_http(args.router_url, "/v1/models"),
        "router_prometheus": check_http(args.router_url, "/metrics/prometheus"),
        "vllm_health": check_http(args.vllm_url, "/health"),
        "vllm_models": check_http(args.vllm_url, "/v1/models"),
    }
    report["ok"] = all(
        section.get("ok", False)
        for key, section in report.items()
        if key not in {"python", "ok"}
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
