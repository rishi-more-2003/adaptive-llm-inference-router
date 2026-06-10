from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStats:
    cuda_available: bool
    memory_allocated_mb: float | None
    memory_reserved_mb: float | None


def get_gpu_stats() -> GpuStats:
    try:
        import torch
    except Exception:
        return GpuStats(False, None, None)

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return GpuStats(False, None, None)

    return GpuStats(
        cuda_available=True,
        memory_allocated_mb=torch.cuda.memory_allocated() / (1024 * 1024),
        memory_reserved_mb=torch.cuda.memory_reserved() / (1024 * 1024),
    )
