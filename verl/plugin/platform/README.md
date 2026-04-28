# verl Platform Abstraction Layer

This package provides a **hardware-agnostic platform interface** so that the
rest of the verl codebase never calls `torch.cuda.*` (or `torch.npu.*`, …)
directly.  Instead, all device-specific logic is routed through a
`PlatformBase` singleton obtained via `get_platform()`.

## Quick Start

```python
from verl.plugin.platform import get_platform

platform = get_platform()            # auto-detected singleton
platform.manual_seed(42)
with platform.nvtx_range("train"):
    ...
```

The platform is **auto-detected** on first call (CUDA → NPU).
Override with the `VERL_PLATFORM` environment variable:

```bash
VERL_PLATFORM=npu python train.py    # force NPU
```

## Package Structure

```
verl/plugin/platform/
├── __init__.py            # Public API: get_platform, set_platform, PlatformRegistry, PlatformBase
├── platform_base.py       # ABC – all methods a backend must implement
├── platform_cuda.py       # NVIDIA CUDA implementation
├── platform_npu.py        # Huawei Ascend NPU implementation
├── platform_manager.py    # PlatformRegistry + singleton manager with auto-detection
└── README.md              # This file
```

## Adding a New Chip / Accelerator

New hardware backends are added via **`@PlatformRegistry.register()`** — no changes to
the verl source tree are required.

### Step 1 — Create a platform class in your plugin package

```python
# my_plugin/platform.py

from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform import PlatformBase, PlatformRegistry


@PlatformRegistry.register(platform="xpu")
class PlatformXPU(PlatformBase):
    """Platform backend for Intel XPU."""

    @property
    def device_name(self) -> str:
        return "xpu"

    @property
    def device_module(self) -> ModuleType:
        return torch.xpu

    def is_available(self) -> bool:
        return torch.xpu.is_available()

    def current_device(self) -> int:
        return torch.xpu.current_device()

    def device_count(self) -> int:
        return torch.xpu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.xpu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.xpu.synchronize(device_index)

    def manual_seed(self, seed: int) -> None:
        torch.xpu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        pass

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    def communication_backend_name(self) -> str:
        return "ccl"

    def visible_devices_envvar(self) -> str:
        return "ZE_AFFINITY_MASK"

    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def cudart(self) -> Any:
        return None
```

### Step 2 — Load via `VERL_USE_EXTERNAL_MODULES`

```bash
export VERL_USE_EXTERNAL_MODULES=my_plugin.platform
```

That's it. The `@PlatformRegistry.register()` decorator registers the platform class
at import time. When `get_platform()` runs auto-detection, it iterates all registered
platforms and picks the first available one. All downstream code (`get_device_name()`,
`get_torch_device()`, etc.) automatically uses the new backend.

## PlatformBase Interface Summary

| Category | Method | Description |
|---|---|---|
| **Device** | `device_name` | Device type string (`'cuda'`, `'npu'`, `'xpu'`, …) |
| | `device_module` | `torch.<device>` namespace module |
| | `is_available()` | Whether the backend is available |
| | `current_device()` | Current device index |
| | `device_count()` | Number of available devices |
| | `set_device(idx)` | Select a device |
| | `synchronize()` | Wait for pending work to complete |
| **RNG** | `manual_seed(seed)` | Seed current device |
| | `manual_seed_all(seed)` | Seed all devices |
| **Memory** | `set_allocator_settings(s)` | Configure memory allocator |
| | `empty_cache()` | Release cached memory |
| **Properties** | `get_device_capability(idx)` | `(major, minor)` or `(None, None)` |
| **Communication** | `communication_backend_name()` | `'nccl'`, `'hccl'`, `'gloo'`, … |
| | `visible_devices_envvar()` | Env var controlling device visibility |
| **Profiling** | `nvtx_range(msg)` | Context manager for profiler ranges |
| | `profiler_start()` | Start device profiler |
| | `profiler_stop()` | Stop device profiler |
| **Low-level** | `cudart()` | CUDA runtime API object or `None` |

## Backward Compatibility

`verl/utils/device.py` is preserved as a thin wrapper.  All existing imports
like `from verl.utils.device import get_device_name` continue to work — they
now delegate to `get_platform()` internally.
