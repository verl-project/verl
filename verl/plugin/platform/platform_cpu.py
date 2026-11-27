# Copyright 2026 BAAI. All rights reserved.
"""CPU platform implementation for accelerator-free workers."""

from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry


@PlatformRegistry.register(platform="cpu")
class PlatformCPU(PlatformBase):
    """Platform backend for CPU-only execution."""

    @property
    def device_name(self) -> str:
        return "cpu"

    @property
    def vendor_name(self) -> str:
        return "cpu"

    @property
    def device_module(self) -> ModuleType:
        return torch.cpu

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def set_device(self, device_index: int) -> None:
        pass

    def synchronize(self, device_index: Optional[int] = None) -> None:
        pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def set_allocator_settings(self, settings: str) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    def communication_backend_name(self) -> str:
        return "gloo"

    def visible_devices_envvar(self) -> str:
        return ""

    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def ray_resource_name(self) -> str:
        return "CPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {}

    def ray_noset_envvars(self) -> list[str]:
        return []

    def is_ipc_supported(self) -> bool:
        return False

    def cudart(self) -> Any:
        return None
