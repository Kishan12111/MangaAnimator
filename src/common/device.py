from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    device: str
    has_gpu: bool
    mixed_precision: str
    batch_size_hint: int
    profile_name: str


def detect_runtime_profile() -> RuntimeProfile:
    """Detect GPU/CPU capabilities with graceful fallback."""
    try:
        import torch

        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_gb >= 35:
                return RuntimeProfile(
                    device="cuda",
                    has_gpu=True,
                    mixed_precision="bf16",
                    batch_size_hint=8,
                    profile_name="colab_a100",
                )
            if total_gb >= 20:
                return RuntimeProfile(
                    device="cuda",
                    has_gpu=True,
                    mixed_precision="fp16",
                    batch_size_hint=4,
                    profile_name="colab_l4",
                )
            return RuntimeProfile(
                device="cuda",
                has_gpu=True,
                mixed_precision="fp16",
                batch_size_hint=2,
                profile_name="colab_t4",
            )
    except Exception:
        pass

    return RuntimeProfile(
        device="cpu",
        has_gpu=False,
        mixed_precision="fp32",
        batch_size_hint=1,
        profile_name="cpu_fallback",
    )
