# Copyright (c) 2026 BAAI. All rights reserved.
"""AMD ROCm/HIP platform implementation.

ROCm is largely CUDA-compatible: PyTorch on ROCm reuses the ``torch.cuda.*``
API surface via hipify, so most of ``PlatformCUDA`` works unchanged. This class
therefore subclasses ``PlatformCUDA`` and only overrides the parts that differ
on ROCm, following a "CUDA base + ROCm extensions" model (always extend the
parent via ``super()`` rather than re-implementing it).
"""

import os
import shutil

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

    def is_platform_available(self, use_smi_check=False) -> bool:
        if not hasattr(torch, "cuda"):
            return False
        # Only ROCm (HIP) torch builds qualify as the AMD platform.
        if torch.version.hip is None:
            return False
        if use_smi_check:
            # In CPU-only Ray actors, torch.cuda.is_available() may return False
            # even though the cluster has GPUs. Fall back to rocm-smi check,
            # and if that's also unavailable (e.g. not on PATH), treat
            # torch.cuda being built as sufficient evidence.
            cmd = "rocm-smi"
            cmd_path = shutil.which(cmd)
            if cmd_path is None:
                # Fallback to common absolute paths if not found in PATH
                common_paths = [
                    f"/usr/bin/{cmd}",
                    f"/usr/local/bin/{cmd}",
                    f"/opt/rocm/bin/{cmd}",
                ]
                for path in common_paths:
                    if os.path.isfile(path) and os.access(path, os.X_OK):
                        cmd_path = path
                        break
                if cmd_path is None:
                    return False
            if self.check_smi_command(cmd_path):
                return True
        return torch.cuda.is_available()

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
