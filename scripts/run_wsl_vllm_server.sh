#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
DTYPE="${DTYPE:-float16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  export CUDA_HOME="${CUDA_HOME:-${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cu13}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

EXTRA_ARGS=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  "${EXTRA_ARGS[@]}"
