# Copyright (c) 2026 BAAI. All rights reserved.
"""Intel GPU (PyTorch XPU) platform implementation."""

import logging
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


def _has_torch_xpu() -> bool:
    """Return True if the ``torch.xpu`` namespace is present."""
    return hasattr(torch, "xpu")


@PlatformRegistry.register(platform="intel")
class PlatformXPU(PlatformBase):
    """Platform backend for Intel GPU (Level Zero / SYCL via ``torch.xpu``)."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "xpu"

    @property
    def vendor_name(self) -> str:
        return "intel"

    @property
    def device_module(self) -> ModuleType:
        return torch.xpu

    def is_available(self) -> bool:
        return _has_torch_xpu() and torch.xpu.is_available()

    def is_platform_available(self, use_smi_check=False) -> bool:
        if not _has_torch_xpu():
            return False
        if use_smi_check:
            # torch.xpu namespace present — Intel GPU environment confirmed even
            # in CPU-only Ray actors where is_available() may report False.
            return True
        return torch.xpu.is_available()

    def current_device(self) -> int:
        return torch.xpu.current_device()

    def device_count(self) -> int:
        return torch.xpu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.xpu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.xpu.synchronize(device_index)

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.xpu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        # torch.xpu does not expose an allocator-settings hook yet; no-op.
        logger.debug("set_allocator_settings is a no-op on Intel GPU: %s", settings)

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # Intel GPU does not expose a CUDA-style (major, minor) compute capability.
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        # oneCCL collective backend for Intel GPU.
        return "xccl"

    def visible_devices_envvar(self) -> str:
        # bare IDs work; ONEAPI_DEVICE_SELECTOR needs a level_zero: prefix.
        return "ZE_AFFINITY_MASK"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        # Ray's IntelGPUAccelerator registers XPU under the "GPU" key, same as CUDA.
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK"]

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        # rebuild_ipc() assumes a CUDA 8-element IPC handle tuple (index 6 = device_id).
        # XPU uses a different SYCL IPC handle format, so the CUDA path would corrupt
        # data. Fall back to shared memory until XPU IPC handle support is added.
        return False

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        # Intel GPU has no NVTX equivalent; log for debugging and yield.
        logger.debug("NVTX range (no-op on Intel GPU): %s", msg)
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None
