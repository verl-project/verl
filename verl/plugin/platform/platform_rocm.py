# Copyright (c) 2026 BAAI. All rights reserved.
"""AMD ROCm/HIP platform implementation.

ROCm is largely CUDA-compatible: PyTorch on ROCm reuses the ``torch.cuda.*``
API surface via hipify, so most of ``PlatformCUDA`` works unchanged. This class
therefore subclasses ``PlatformCUDA`` and only overrides the parts that differ
on ROCm, following a "CUDA base + ROCm extensions" model (always extend the
parent via ``super()`` rather than re-implementing it).
"""

import os

import torch

from .platform_cuda import PlatformCUDA
from .platform_manager import PlatformRegistry


@PlatformRegistry.register(platform="amd")
class PlatformROCm(PlatformCUDA):
    """Platform backend for AMD ROCm/HIP GPUs (reuses PlatformCUDA where compatible)."""

    @property
    def vendor_name(self) -> str:
        # NOTE: device_name stays 'cuda' on purpose — PyTorch ROCm exposes the
        # device type string as "cuda" (torch.device("cuda") works via hipify).
        return "amd"

    def is_available(self) -> bool:
        return torch.cuda.is_available() and torch.version.hip is not None

    def is_platform_available(self, use_smi_check=False) -> bool:
        # ROCm is identified by torch being a HIP build; does not depend on
        # nvidia-smi / rocm-smi being on PATH.
        return torch.version.hip is not None

    def rollout_env_vars(self) -> dict[str, str]:
        # Extend CUDA's rollout env vars with ROCm-specific ones. SGLANG_USE_AITER
        # routes SGLang's non-attention kernels (RMSNorm/RoPE/MoE/quant) through AITER.
        # Default to "1" but honor an explicit user override (e.g. SGLANG_USE_AITER=0
        # to fall back to vLLM kernels).
        return {
            **super().rollout_env_vars(),
            "SGLANG_USE_AITER": os.environ.get("SGLANG_USE_AITER", "1"),
        }

    def ray_noset_envvars(self) -> list[str]:
        # On ROCm, HIP_VISIBLE_DEVICES takes precedence over CUDA_VISIBLE_DEVICES,
        # and ROCR_VISIBLE_DEVICES is also relevant, so tell Ray not to manage them.
        return super().ray_noset_envvars() + [
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
            "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        ]
