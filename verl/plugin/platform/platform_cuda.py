# Copyright (c) 2026 BAAI. All rights reserved.
# Adopted from https://github.com/microsoft/DeepSpeed/blob/master/accelerator/cuda_accelerator.py
"""NVIDIA CUDA platform implementation."""

import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch
import torch.cuda

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry


@PlatformRegistry.register(platform="cuda")
class PlatformCUDA(PlatformBase):
    """Platform backend for NVIDIA CUDA GPUs."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "cuda"

    @property
    def device_module(self) -> ModuleType:
        return torch.cuda

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def set_device(self, device_index: int) -> None:
        torch.cuda.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.cuda.synchronize(device_index)

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        torch.cuda.memory._set_allocator_settings(settings)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return torch.cuda.get_device_capability(device_index)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "flagcx" if os.getenv("USE_FLAGCX", "0").lower() in ["1", "true"] else "nccl"

    def visible_devices_envvar(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        with torch.cuda.nvtx.range(msg):
            yield

    def profiler_start(self) -> None:
        torch.cuda.profiler.start()

    def profiler_stop(self) -> None:
        torch.cuda.profiler.stop()

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return torch.cuda.cudart()
