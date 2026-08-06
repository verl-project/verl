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

The platform is **auto-detected** on first call. Detection order:

1. `VERL_PLATFORM` environment variable (explicit override)
2. Probe all registered platforms via `is_available(use_smi_check=True)`
3. Fall back to `cuda`

```bash
VERL_PLATFORM=npu python train.py      # force NPU
VERL_PLATFORM=metax python train.py    # force MetaX
```

## Package Structure

```
verl/plugin/platform/
├── __init__.py            # Public API: get_platform, set_platform, PlatformRegistry, PlatformBase
├── platform_base.py       # ABC – all methods a backend must implement
├── platform_cuda.py       # NVIDIA CUDA implementation (built-in)
├── platform_npu.py        # Huawei Ascend NPU implementation (built-in)
├── platform_manager.py    # PlatformRegistry + singleton manager with auto-detection
└── README.md              # This file
```

## Design Principles

### Registration uses platform name only (no vendor)

Each platform is registered with a unique **platform name** (e.g. `cuda`, `npu`,
`metax`, `xpu`, `mlu`). The platform name itself is already vendor-distinguishing,
so the registry does not require a separate vendor parameter:

```python
@PlatformRegistry.register(platform="metax")    # platform name is the key
class PlatformMetaX(PlatformBase):
    ...
```

### Platform instances carry vendor info

While registration/lookup only uses the platform name, each platform instance
exposes a `vendor_name` property for informational and display purposes:

```python
platform = get_platform()
print(platform.vendor_name)   # "nvidia", "metax", "huawei", "intel", "cambricon", ...
```

### Model patches are platform-dispatched

Platforms can override `apply_model_patches(model_type)` to apply device-specific
monkey patches (e.g. replacing RMSNorm/RoPE with NPU-optimized ops). The engine
calls this hook during initialization, before model construction.

## Adding a New Chip / Accelerator

New hardware backends are added via **`@PlatformRegistry.register()`** — no changes to
the verl source tree are required.

### Step 1 — Create a platform class in your plugin package

```python
# my_plugin/platform_xpu.py

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
    def vendor_name(self) -> str:
        return "intel"

    @property
    def device_module(self) -> ModuleType:
        return torch.xpu

    def is_available(self, use_smi_check=False) -> bool:
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
        return "xccl"

    def visible_devices_envvar(self) -> str:
        return "ZE_AFFINITY_MASK"

    def ray_resource_name(self) -> str:
        return "GPU"

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK"]

    def is_ipc_supported(self) -> bool:
        return False

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

### Step 2 — Load via `VERL_USE_EXTERNAL_MODULES` or entry_points

**Option A: Environment variable (explicit control)**

```bash
export VERL_USE_EXTERNAL_MODULES=my_plugin.platform_xpu
python train.py
```

**Option B: setuptools entry_points (auto-discovery)**

In your plugin's `pyproject.toml`:

```toml
[project.entry-points."verl.plugins"]
my_hardware = "my_plugin.platform_xpu"
```

When `import verl` runs, it auto-discovers all packages registered under the
`verl.plugins` entry_points group and loads them. The `@PlatformRegistry.register()`
decorator fires at import time, making the platform available for auto-detection.

Control plugin loading via `VERL_USE_EXTERNAL_PLUGINS`:

```bash
VERL_USE_EXTERNAL_PLUGINS=auto       # load all (default)
VERL_USE_EXTERNAL_PLUGINS=none       # disable discovery
VERL_USE_EXTERNAL_PLUGINS=pkg1,pkg2  # only load named entry_points
```

## PlatformRegistry API

| Method | Description |
|---|---|
| `PlatformRegistry.register(platform="name")` | Class decorator to register a platform |
| `PlatformRegistry.get("name")` | Look up a platform class by name |
| `PlatformRegistry.registered_names()` | Tuple of all registered platform names |

## PlatformBase Interface Summary

| Category | Method / Property | Description |
|---|---|---|
| **Identity** | `device_name` | Device type string (`'cuda'`, `'npu'`, `'xpu'`, …) |
| | `vendor_name` | Hardware vendor (`'nvidia'`, `'huawei'`, `'intel'`, …) |
| **Device** | `device_module` | `torch.<device>` namespace module |
| | `is_available(use_smi_check)` | Whether the backend is available |
| | `current_device()` | Current device index |
| | `device_count()` | Number of available devices |
| | `set_device(idx)` | Select a device |
| | `synchronize()` | Wait for pending work to complete |
| **RNG** | `manual_seed(seed)` | Seed current device |
| | `manual_seed_all(seed)` | Seed all devices |
| **Memory** | `set_allocator_settings(s)` | Configure memory allocator |
| | `empty_cache()` | Release cached memory |
| **Properties** | `get_device_capability(idx)` | `(major, minor)` or `(None, None)` |
| **Communication** | `communication_backend_name()` | `'nccl'`, `'hccl'`, `'xccl'`, … |
| | `visible_devices_envvar()` | Env var controlling device visibility |
| | `get_collective_module()` | Collective comm module (e.g. `cupy.cuda.nccl`) |
| **Ray** | `ray_resource_name()` | Ray resource name (`'GPU'`, `'NPU'`, …) |
| | `ray_noset_envvars()` | `RAY_EXPERIMENTAL_NOSET_*` env var names |
| | `ray_resource_options(num_gpus)` | Ray actor resource dict |
| **IPC** | `is_ipc_supported()` | Whether IPC tensor sharing is supported |
| **Rollout** | `rollout_env_vars()` | Env vars for rollout engine launch |
| **Model Patches** | `apply_model_patches(model_type)` | Apply platform-specific model monkey patches |
| **Profiling** | `nvtx_range(msg)` | Context manager for profiler ranges |
| | `profiler_start()` | Start device profiler |
| | `profiler_stop()` | Stop device profiler |
| **Low-level** | `cudart()` | CUDA runtime API object or `None` |
| | `check_smi_command(cmd)` | Run SMI command and check exit code |

## Environment Variables

| Variable | Description |
|---|---|
| `VERL_PLATFORM` | Override auto-detection (e.g. `cuda`, `npu`, `metax`, `xpu`, `mlu`) |
| `VERL_USE_EXTERNAL_PLUGINS` | Control entry_points discovery: `auto` (default), `none`, or comma-separated names |

## Backward Compatibility

`verl/utils/device.py` is preserved as a thin wrapper. All existing imports
like `from verl.utils.device import get_device_name` continue to work — they
now delegate to `get_platform()` internally.

Key utility functions in `verl/utils/device.py`:

- `get_device_name()` → `get_platform().device_name`
- `get_vendor()` → `get_platform().vendor_name`
- `get_visible_devices_keyword()` → `get_platform().visible_devices_envvar()`
- `get_resource_name()` → `get_platform().ray_resource_name()`
