from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ComputeSnapshot:
    device: str
    gpu_name: str
    total_vram_gb: float
    used_vram_gb: float
    free_vram_gb: float


def get_compute_snapshot() -> ComputeSnapshot:
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            free, total = torch.cuda.mem_get_info(idx)
            used = total - free
            return ComputeSnapshot(
                device="cuda",
                gpu_name=props.name,
                total_vram_gb=round(total / (1024**3), 2),
                used_vram_gb=round(used / (1024**3), 2),
                free_vram_gb=round(free / (1024**3), 2),
            )
    except Exception:
        pass

    return ComputeSnapshot(device="cpu", gpu_name="none", total_vram_gb=0.0, used_vram_gb=0.0, free_vram_gb=0.0)


def gpu_warmup(size: int = 2048) -> dict[str, Any]:
    """Use GPU math kernel warmup to ensure CUDA path is active in Colab."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"warmup": False, "reason": "cuda_unavailable"}

        x = torch.randn((size, size), device="cuda", dtype=torch.float16)
        y = torch.randn((size, size), device="cuda", dtype=torch.float16)
        z = x @ y
        _ = float(z.mean().item())
        torch.cuda.synchronize()
        return {"warmup": True, "device": "cuda", "matrix": size}
    except Exception as exc:
        return {"warmup": False, "reason": str(exc)}
