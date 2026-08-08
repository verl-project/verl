# Copyright 2024-2025 BAAI and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Google TPU platform implementation.

TPU with PyTorch/XLA (torch_tpu) reuses the ``torch.cuda.*`` API surface, so most of
``PlatformCUDA`` works unchanged. This class subclasses ``PlatformCUDA`` and overrides
device-specific environment configuration, resource options, and memory management proxies.
"""

import logging
import os
from typing import Any

import ray
import torch

from .platform_cuda import PlatformCUDA
from .platform_manager import PlatformRegistry, get_platform

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Communication port defaults for TPU distributed slice builder meshes
ROLLOUT_BASE_PORT = 8070
TRAINER_BASE_PORT = 8471

# TPU Chip HBM capacities in bytes
HBM_BYTES_TPU_V5P = 95 * 1024 * 1024 * 1024  # 95 GB
HBM_BYTES_TPU_V6E = 32 * 1024 * 1024 * 1024  # 32 GB
HBM_BYTES_TPU_V7X = 192 * 1024 * 1024 * 1024  # 192 GB


def get_tpu_chip_hbm_bytes() -> int:
    """Detects the TPU chip generation from Ray node labels or environment variables and returns its HBM capacity."""
    tpu_type = ""

    # Query Ray cluster node labels for TPU resource type
    try:
        if ray.is_initialized():
            tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
            if tpu_nodes:
                labels = tpu_nodes[0].get("Labels", {})
                tpu_type = (labels.get("ray.io/accelerator-type") or labels.get("ray.io/tpu-pod-type") or "").lower()
    except Exception as e:
        logger.debug(f"Unable to query Ray node labels for TPU chip type: {e}")

    # Fallback to environment variables
    if not tpu_type:
        tpu_type = (
            os.environ.get("TPU_ACCELERATOR_TYPE")
            or os.environ.get("ACCELERATOR_TYPE")
            or os.environ.get("TPU_TYPE")
            or ""
        ).lower()

    if "v5p" in tpu_type:
        return HBM_BYTES_TPU_V5P
    elif "v6e" in tpu_type:
        return HBM_BYTES_TPU_V6E
    elif "v7x" in tpu_type:
        return HBM_BYTES_TPU_V7X
    else:
        logger.warning(f"Unable to determine TPU chip HBM bytes for tpu_type='{tpu_type}'. Returning -1.")
        return -1


# Enforce static compilation graph for torch.compile on TPU
try:
    _orig_compile = torch.compile

    def patched_compile(*args, **kwargs):
        if get_platform().device_name == "tpu":
            kwargs["dynamic"] = False
        return _orig_compile(*args, **kwargs)

    torch.compile = patched_compile
except Exception as e:
    logger.warning(f"Failed to patch torch.compile for TPU: {e}")


class DummyTpuDeviceModule:
    """Fallback device module for CPU-only nodes and driver processes.

    Provides no-op implementations for torch.tpu APIs on processes where torch_tpu is not imported
    or no TPU devices are attached.
    """

    def is_available(self) -> bool:
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 0

    def synchronize(self) -> None:
        pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)


class TPUDeviceModuleProxy:
    """Proxy wrapper for torch.tpu to emulate PyTorch CUDA memory management APIs.

    Provides default fallback implementations for CUDA memory tracking methods
    (e.g., memory_reserved, memory_allocated, get_device_properties) that are called throughout verl's codebase
    but not natively provided by torch_tpu.
    """

    def __init__(self, original_module):
        self.__dict__["_original_module"] = original_module

    def __getattr__(self, name):
        if name == "set_device":
            return self.set_device
        elif name in ("memory_reserved", "memory_allocated", "max_memory_reserved", "max_memory_allocated"):
            return lambda *args, **kwargs: 0
        elif name in ("reset_peak_memory_stats", "reset_accumulated_memory_stats"):
            return lambda *args, **kwargs: None
        elif name == "empty_cache":
            return self.empty_cache
        elif name == "get_device_properties":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            return lambda device_index=None: type("DeviceProps", (), {"total_memory": hbm_bytes})()
        elif name == "mem_get_info":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            return lambda *args, **kwargs: (hbm_bytes, hbm_bytes)
        return getattr(self._original_module, name)

    def set_device(self, device_index: Any) -> None:
        pass

    def empty_cache(self) -> None:
        if hasattr(self._original_module, "_clear_cache"):
            try:
                self._original_module._clear_cache()
            except Exception as e:
                logger.warning(f"Failed to clear TPU cache: {e}")


@PlatformRegistry.register(platform="tpu")
class PlatformTPU(PlatformCUDA):
    """Platform backend for Google TPUs (subclasses PlatformCUDA for API compatibility)."""

    @property
    def vendor_name(self) -> str:
        return "google"

    @property
    def device_name(self) -> str:
        return "tpu"

    @property
    def device_module(self):
        original_tpu = getattr(torch, "tpu", DummyTpuDeviceModule())
        return TPUDeviceModuleProxy(original_tpu)

    def current_device(self) -> int:
        return self.device_module.current_device()

    def device_count(self) -> int:
        return self.device_module.device_count()

    def set_device(self, device_index: int) -> None:
        self.device_module.set_device(device_index)

    def synchronize(self, device_index: int | None = None) -> None:
        self.device_module.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        self.device_module.manual_seed_all(seed)

    def is_available(self) -> bool:
        if hasattr(torch, "tpu"):
            try:
                return torch.tpu.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
        return False

    def is_platform_available(self, use_smi_check=False) -> bool:
        if os.environ.get("VERL_PLATFORM") == "tpu":
            return True
        if "TPU_NAME" in os.environ or "TPU_VISIBLE_DEVICES" in os.environ:
            return True
        return False

    def ray_resource_name(self) -> str:
        return "TPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        tpu_chips = int(num_gpus)
        return {"resources": {"TPU": tpu_chips}} if tpu_chips >= 1 else {}

    def communication_backend_name(self) -> str:
        return "tpu_dist"

    def ray_noset_envvars(self) -> list[str]:
        return super().ray_noset_envvars() + [
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        ]

    def get_tpu_env_vars(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        pgs: list,
    ) -> dict[str, str]:
        """Generates TPU-specific distributed environment variables for PJRT mesh initialization."""
        node_ip_map = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
        pg_ips = []
        for pg in pgs:
            specs = ray._private.state.state.placement_group_table(pg.id)
            node_id = specs["bundles_to_node_id"][0]
            pg_ips.append(node_ip_map[node_id])

        is_rollout = "rollout" in name_prefix.lower()
        base_port = ROLLOUT_BASE_PORT if is_rollout else TRAINER_BASE_PORT

        sb_addresses = []
        for ip in pg_ips:
            for lr in range(local_world_size):
                sb_addresses.append(f"{ip}:{base_port + lr}")

        env_vars = {
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "CLOUD_TPU_TASK_ID": str(rank // local_world_size),
            "TPU_WORKER_HOSTNAMES": ",".join(pg_ips),
            "TPU_VISIBLE_CHIPS": str(local_rank),
        }

        # Apply TPU topology and host bounds based on TPU pod type or world size
        tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
        tpu_type = tpu_nodes[0].get("Labels", {}).get("ray.io/tpu-pod-type", "") if tpu_nodes else ""

        if tpu_type == "v6e-32":
            topo = "4,8,1"
        elif tpu_type == "v6e-8":
            topo = "2,4,1"
        elif tpu_type == "v6e-4":
            topo = "2,2,1"
        else:
            topo = (
                "4,8,1"
                if world_size == 32
                else ("2,4,1" if world_size == 8 else ("2,2,1" if world_size == 4 else "1,1,1"))
            )

        env_vars.update(
            {
                "TORCH_TPU_TOPOLOGY": topo,
                "TPU_HOST_BOUNDS": topo,
                "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
                "CHIPS_PER_HOST": "4",
            }
        )

        if is_rollout:
            env_vars.update(
                {
                    "SKIP_JAX_PRECOMPILE": "1",
                    "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
                }
            )
            if world_size > 1:
                env_vars["TPU_MULTIHOST_BACKEND"] = "ray"

        return env_vars
