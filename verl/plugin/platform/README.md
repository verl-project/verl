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
├── __init__.py            # Public API: get_platform, set_platform, PlatformBase
├── platform_base.py       # ABC – all methods a backend must implement
├── platform_cuda.py       # NVIDIA CUDA implementation
├── platform_npu.py        # Huawei Ascend NPU implementation
├── platform_manager.py    # Singleton manager with auto-detection
└── README.md              # This file
```

## Adding a New Chip / Accelerator

To add support for a new backend (e.g. Intel XPU, AMD ROCm, Cambricon MLU):

### Step 1 — Create the platform file

Create `verl/plugin/platform/platform_xpu.py` (replace `xpu` with your backend
name):

```python
# Copyright (c) BAAI Corporation.

from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase


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

    # -- RNG --
    def manual_seed(self, seed: int) -> None:
        torch.xpu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)

    # -- Memory --
    def set_allocator_settings(self, settings: str) -> None:
        pass  # XPU may not support this

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()

    # -- Device properties --
    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)  # XPU uses a different capability model

    # -- Communication --
    def communication_backend_name(self) -> str:
        return "ccl"  # Intel oneCCL

    def visible_devices_envvar(self) -> str:
        return "ZE_AFFINITY_MASK"

    # -- Profiling --
    @contextmanager
    def nvtx_range(self, msg: str):
        yield  # no-op or use Intel ITT

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # -- Low-level --
    def cudart(self) -> Any:
        return None
```

### Step 2 — Register in the platform manager

Edit `platform_manager.py` and add two things:

**a) Auto-detection** (in `_detect_platform_name()`):

```python
# Auto-detect Intel XPU
try:
    import torch
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
except (ImportError, RuntimeError):
    pass
```

**b) Instantiation** (in `_create_platform()`):

```python
if name == "xpu":
    from .platform_xpu import PlatformXPU
    return PlatformXPU()
```

### Step 3 — Done

No other files need modification.  All existing code that uses
`get_platform()`, `get_device_name()`, `get_torch_device()`, etc. will
automatically pick up the new backend.

### Alternative: Runtime registration (no file edits)

If you prefer not to modify `platform_manager.py`, you can register a custom
platform at startup before any other verl code runs:

```python
from verl.plugin.platform import set_platform
from my_backend import MyPlatform

set_platform(MyPlatform())
```

## PlatformBase Interface Summary

| Category | Method | Description |
|---|---|---|
| **Device** | `device_name` | Device type string (`'cuda'`, `'npu'`, `'cpu'`, …) |
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
