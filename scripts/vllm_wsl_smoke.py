from __future__ import annotations

import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    started = time.perf_counter()
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "512")),
        dtype=os.getenv("DTYPE", "float16"),
        enforce_eager=os.getenv("ENFORCE_EAGER", "1") == "1",
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.70")),
    )
    outputs = llm.generate(
        ["Write code for a tiny Python p95 helper."],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    payload = {
        "ok": True,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "text": outputs[0].outputs[0].text,
    }
    output_path = Path("/mnt/c/Data/GithubRepository/adaptive-llm-inference-router/runs/vllm_wsl_direct_smoke.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
