"""AWS Neuron platform implementation"""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


def _ensure_torch_neuron() -> bool:
    """Try to import torch_neuronx so that torch.neuron becomes available.

    Returns True if torch.neuron is usable after the attempt.
    """
    if hasattr(torch, "neuron"):
        return True
    try:
        import torch_neuronx  # noqa: F401

        return hasattr(torch, "neuron")
    except Exception as e:
        logger.debug("The current machine has no torch.neuron, because: %s", e)
    return False


_ensure_torch_neuron()  # Attempt to import torch_neuronx at module load time so that availability checks are faster later


@PlatformRegistry.register(platform="aws")
class PlatformNeuron(PlatformBase):
    """Platform backend for AWS Neuron"""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "neuron"

    @property
    def vendor_name(self) -> str:
        return "aws"

    @property
    def device_module(self) -> ModuleType:
        return torch.neuron

    def is_available(self) -> bool:
        return torch.neuron.is_available()

    def is_platform_available(self, use_smi_check=False) -> bool:
        """Return True if this platform is available on this host.

        Used during auto-detection to determine if the environment targets
        this platform.  When ``use_smi_check=True``, only requires that
        torch_neuron is importable (even if no devices are visible).
        """
        if not _ensure_torch_neuron():
            return False
        if use_smi_check:
            return hasattr(torch, "neuron")
        return torch.neuron.is_available()

    def current_device(self) -> int:
        return torch.neuron.current_device()

    def device_count(self) -> int:
        return torch.neuron.device_count()

    def set_device(self, device_index: int) -> None:
        torch.neuron.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.neuron.synchronize(device_index)

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.neuron.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.neuron.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        # no neuron implementation for allocator settings
        pass

    def empty_cache(self) -> None:
        torch.neuron.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0):
        if hasattr(torch.neuron, "get_device_capability"):
            result = torch.neuron.get_device_capability(device_index)
            # torch.neuron.get_device_capability may return None instead of a tuple
            if result is None:
                return (None, None)
            return result
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "neuron"

    def visible_devices_envvar(self) -> str:
        return "NEURON_RT_VISIBLE_CORES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        return "neuron_cores"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"resources": {"neuron_cores": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES"]

    def rollout_env_vars(self) -> dict[str, str]:
        return {}

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        # Neuron does not have an NVTX equivalent, but we log for debugging
        logger.debug("NVTX range (no-op on Neuron): %s", msg)
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