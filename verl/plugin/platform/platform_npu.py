# Copyright (c) 2026 BAAI. All rights reserved.
"""Huawei Ascend NPU platform implementation."""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


@PlatformRegistry.register(platform="npu")
class PlatformNPU(PlatformBase):
    """Platform backend for Huawei Ascend NPU."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "npu"

    @property
    def device_module(self) -> ModuleType:
        return torch.npu

    def is_available(self) -> bool:
        return torch.npu.is_available()

    def current_device(self) -> int:
        return torch.npu.current_device()

    def device_count(self) -> int:
        return torch.npu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.npu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.npu.synchronize(device_index)

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.npu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.npu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        try:
            torch.npu.memory._set_allocator_settings(settings)
        except Exception:
            logger.warning(
                "Current version of torch-npu does not support `_set_allocator_settings`, "
                "please upgrade torch-npu to 2.9.0 or later"
            )

    def empty_cache(self) -> None:
        torch.npu.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        if hasattr(torch.npu, "get_device_capability"):
            return torch.npu.get_device_capability(device_index)
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "flagcx" if os.getenv("USE_FLAGCX", "0").lower() in ["1", "true"] else "hccl"

    def visible_devices_envvar(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        # NPU does not have an NVTX equivalent, but we log for debugging
        logger.debug("NVTX range (no-op on NPU): %s", msg)
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
