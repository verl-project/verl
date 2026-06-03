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


def _ensure_torch_npu() -> bool:
    """Try to import torch_npu so that torch.npu becomes available.

    Returns True if torch.npu is usable after the attempt.
    """
    if hasattr(torch, "npu"):
        return True
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu")
    except ImportError:
        return False


@PlatformRegistry.register(platform="huawei")
class PlatformNPU(PlatformBase):
    """Platform backend for Huawei Ascend NPU."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "npu"

    @property
    def vendor_name(self) -> str:
        return "huawei"

    @property
    def device_module(self) -> ModuleType:
        if not _ensure_torch_npu():
            raise RuntimeError("torch_npu is not installed or torch.npu is not available")
        return torch.npu

    def is_available(self, use_smi_check=False) -> bool:
        # For NPU, we rely on torch.npu.is_available() which checks both the presence of the driver and the device.
        if not _ensure_torch_npu():
            return False
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
            result = torch.npu.get_device_capability(device_index)
            # torch.npu.get_device_capability may return None instead of a tuple
            if result is None:
                return (None, None)
            return result
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "flagcx" if os.getenv("USE_FLAGCX", "0").lower() in ["1", "true"] else "hccl"

    def visible_devices_envvar(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        return "NPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"resources": {"NPU": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"]

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        import subprocess

        from verl.utils.device import check_ipc_version_support, get_npu_versions

        try:
            software_version, cann_version = get_npu_versions()
            return check_ipc_version_support(software_version, cann_version)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute npu-smi command: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error checking IPC support: {e}") from e

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
