# Copyright (c) 2026 BAAI. All rights reserved.
# Copyright (c) 2026 Google LLC. All rights reserved.
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

**Experimental: TPU support is under development and cannot run training end to end yet.**
This module registers the platform and provides the device abstraction plus the Ray resource
and worker-environment plumbing. The training engine, the generation engine and the
weight-transfer path are still to come.

``torch_tpu`` reuses the NVIDIA device API surface, so most of ``PlatformCUDA`` works
unchanged. This class subclasses it and overrides device-specific environment configuration,
resource options, and memory management proxies.
"""

import logging
import os
from typing import Any, Optional

import ray
import torch

from .platform_cuda import PlatformCUDA
from .platform_manager import PlatformRegistry

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _ensure_torch_tpu() -> bool:
    """Try to import torch_tpu, which registers the ``tpu_dist`` distributed backend.

    Returns True if the TPU runtime is usable after the attempt.
    """
    if hasattr(torch, "tpu"):
        return True
    try:
        import torch_tpu  # noqa: F401

        return hasattr(torch, "tpu")
    except Exception as e:
        logger.debug("The current machine has no torch.tpu, because: %s", e)
    return False


_ensure_torch_tpu()  # Attempt at module load time so availability checks are faster later

# Base port for the TPU distributed slice builder mesh. Each local rank takes ``base + local_rank``.
TPU_PROCESS_BASE_PORT = 8471

# TPU chip HBM capacities in bytes
HBM_BYTES_TPU_V5P = 95 * 1024 * 1024 * 1024  # 95 GB
HBM_BYTES_TPU_V6E = 32 * 1024 * 1024 * 1024  # 32 GB
HBM_BYTES_TPU_V7X = 192 * 1024 * 1024 * 1024  # 192 GB

TPU_HBM_BYTES_MAP = {
    "v5p": HBM_BYTES_TPU_V5P,
    "v6e": HBM_BYTES_TPU_V6E,
    "v7x": HBM_BYTES_TPU_V7X,
}

# Fallback HBM capacity when the chip generation cannot be determined.
HBM_BYTES_TPU_DEFAULT = HBM_BYTES_TPU_V6E

# TPU default 3D mesh topology mappings by pod type or total chips
TPU_TOPOLOGY_MAP = {
    "v6e-32": "4,8,1",
    "v6e-8": "2,4,1",
    "v6e-4": "2,2,1",
    32: "4,8,1",
    8: "2,4,1",
    4: "2,2,1",
}


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
        logger.warning(f"Unable to query Ray node labels for TPU chip type: {e}")

    # Fallback to environment variables
    if not tpu_type:
        tpu_type = (
            os.environ.get("TPU_ACCELERATOR_TYPE")
            or os.environ.get("ACCELERATOR_TYPE")
            or os.environ.get("TPU_TYPE")
            or ""
        ).lower()

    for chip_gen, hbm_bytes in TPU_HBM_BYTES_MAP.items():
        if chip_gen in tpu_type:
            return hbm_bytes

    logger.warning(f"Unable to determine TPU chip HBM bytes for tpu_type='{tpu_type}'. Returning -1.")
    return -1


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
    """Proxy wrapper for torch.tpu to emulate the PyTorch NVIDIA memory management APIs.

    Provides default fallback implementations for memory tracking methods
    (e.g. memory_reserved, memory_allocated, get_device_properties) that are called throughout verl's codebase
    but not natively provided by torch_tpu.
    """

    def __init__(self, original_module):
        self.__dict__["_original_module"] = original_module

    def __getattr__(self, name):
        if name == "set_device":
            return self.set_device

        if hasattr(self._original_module, name):
            return getattr(self._original_module, name)

        if name == "memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "reset_peak_memory_stats":
            return lambda *args, **kwargs: None
        elif name == "get_device_properties":

            class DummyDeviceProperties:
                def __init__(self, total_memory=HBM_BYTES_TPU_DEFAULT):
                    self.total_memory = total_memory
                    self.name = "Google TPU"
                    self.major = 1
                    self.minor = 0

            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else HBM_BYTES_TPU_DEFAULT
            return lambda *args, **kwargs: DummyDeviceProperties(total_memory=total_mem)
        elif name == "mem_get_info":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else HBM_BYTES_TPU_DEFAULT
            return lambda *args, **kwargs: (total_mem, total_mem)

        raise AttributeError(f"'TPUDeviceModuleProxy' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._original_module, name, value)

    def is_available(self) -> bool:
        if hasattr(self._original_module, "is_available"):
            try:
                return self._original_module.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
                return False
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        if hasattr(self._original_module, "current_device"):
            try:
                return self._original_module.current_device()
            except Exception as e:
                logger.warning(f"torch.tpu.current_device() failed: {e}")
                return 0
        return 0

    def device_count(self) -> int:
        if hasattr(self._original_module, "device_count"):
            try:
                return self._original_module.device_count()
            except Exception as e:
                logger.warning(f"torch.tpu.device_count() failed: {e}")
                return 0
        return 0

    def synchronize(self, device_index: int | None = None) -> None:
        if hasattr(self._original_module, "synchronize"):
            try:
                self._original_module.synchronize()
            except Exception as e:
                logger.warning(f"torch.tpu.synchronize() failed: {e}")

    def empty_cache(self) -> None:
        if hasattr(self._original_module, "_clear_cache"):
            try:
                self._original_module._clear_cache()
            except Exception as e:
                logger.warning(f"Failed to clear TPU cache: {e}")


@PlatformRegistry.register(platform="tpu")
class PlatformTPU(PlatformCUDA):
    """Platform backend for Google TPUs (subclasses PlatformCUDA for API compatibility).

    Experimental: see the module docstring. Registering this platform does not make verl
    runnable on TPU on its own -- the engine backends are still to come.
    """

    def __init__(self):
        super().__init__()
        original_tpu = getattr(torch, "tpu", DummyTpuDeviceModule())
        self._device_module = TPUDeviceModuleProxy(original_tpu)

    @property
    def vendor_name(self) -> str:
        return "google"

    @property
    def device_name(self) -> str:
        return "tpu"

    @property
    def device_module(self):
        return self._device_module

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
        # The env vars come first because a CPU-only Ray driver process has no chip attached
        # yet still belongs to a TPU job. Falling back to the runtime lets a plain TPU VM,
        # where nothing sets TPU_NAME, detect itself instead of defaulting to CUDA.
        return self.is_available()

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

    def ray_local_rank_override(self) -> Optional[str]:
        # Ray does not enumerate TPU chips, so the index comes from TPU_VISIBLE_CHIPS, which
        # get_worker_env_vars() sets when the worker is created.
        return os.environ.get("TPU_VISIBLE_CHIPS", "0")

    def supports_colocated_worker_groups(self) -> bool:
        # A TPU chip belongs to one process and cannot be shared between colocated WorkerGroups.
        return False

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
        node_ip_map = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes() if node.get("Alive", False)}
        bundle_ips = []
        local_ip = ray.util.get_node_ip_address()
        clean_prefix = name_prefix.lower().split("_")[0] if name_prefix else ""
        matching_pgs = []

        # 1. Primary filter: Select placement group containing current worker's node IP
        for p in pgs:
            specs = ray._private.state.state.placement_group_table(p.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            pg_ips = [node_ip_map[node_id] for b_idx, node_id in sorted(bundles_map.items()) if node_id in node_ip_map]
            if local_ip in pg_ips:
                matching_pgs.append(p)

        # 2. Secondary fallback: Filter by clean_prefix if placement group names are explicitly set
        if not matching_pgs and clean_prefix:
            for p in pgs:
                p_name = ray._private.state.state.placement_group_table(p.id).get("name", "").lower()
                if clean_prefix in p_name:
                    matching_pgs.append(p)

        target_pgs = matching_pgs if matching_pgs else pgs

        for pg in target_pgs:
            specs = ray._private.state.state.placement_group_table(pg.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            for b_idx in sorted(bundles_map.keys()):
                node_id = bundles_map[b_idx]
                if node_id in node_ip_map:
                    bundle_ips.append(node_ip_map[node_id])

        base_port = TPU_PROCESS_BASE_PORT
        sb_addresses = [f"{ip}:{base_port + (b_idx % local_world_size)}" for b_idx, ip in enumerate(bundle_ips)]

        # Extract unique worker hostnames preserving rank order
        unique_hostnames = list(dict.fromkeys(bundle_ips))

        env_vars = {
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "CLOUD_TPU_TASK_ID": str(rank // local_world_size),
            "TPU_WORKER_HOSTNAMES": ",".join(unique_hostnames),
            "TPU_VISIBLE_CHIPS": str(local_rank),
        }

        # Apply TPU topology and host bounds based on TPU pod type or world size
        tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
        tpu_type = tpu_nodes[0].get("Labels", {}).get("ray.io/tpu-pod-type", "") if tpu_nodes else ""

        topo = TPU_TOPOLOGY_MAP.get(tpu_type, TPU_TOPOLOGY_MAP.get(world_size, "1,1,1"))

        env_vars.update(
            {
                "TORCH_TPU_TOPOLOGY": topo,
                "TPU_HOST_BOUNDS": topo,
                "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
                "CHIPS_PER_HOST": "4",
            }
        )

        return env_vars

    def get_worker_env_vars(
        self,
        resource_pool,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        device_name: str,
    ) -> dict[str, str]:
        """Return platform-specific TPU environment variables for worker nodes."""
        env_vars = {}
        if "VERL_PLATFORM" in os.environ:
            env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]
        for var in self.ray_noset_envvars():
            env_vars[var] = "1"
        pgs = resource_pool.get_placement_groups(device_name=device_name)
        tpu_env = self.get_tpu_env_vars(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            name_prefix=name_prefix,
            pgs=pgs,
        )
        env_vars.update(tpu_env)
        return env_vars
