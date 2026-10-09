# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, cast

from omegaconf import DictConfig
from pydantic import BaseModel

from verl.runtime import ClassWithInitArgs, RemoteWorkerGroup, ResourcePool, Worker, WorkerGroup, current_runtime
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig

logger = logging.getLogger(__file__)


# Max number of concurrent calls to the methods of Rollout,
# excluding calls to generate method.
CONTROL_METHOD_CONCURRENCY = 16


class TokenOutput(BaseModel):
    token_ids: list[int]
    """response token ids"""
    log_probs: Optional[list[float]] = None
    """logprobs of response token ids"""
    routed_experts: Optional[Any] = None
    """routed experts of response token ids"""
    stop_reason: Optional[str] = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    num_preempted: Optional[int] = None
    """number of preempted times for metric calculation"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class RolloutMode(Enum):
    # Rollout engine and training engine(fsdp/megatron) fused in same process
    # Rollout and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Rollout engine colocated with hybrid engine in same ray placement group but in separate process.
    # Rollout and hybrid processes share GPUs, switch context without weight synchronization.
    # Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone rollout server with separate GPU resource, disaggregated architecture.
    # Usage scenarios: off-policy training.
    STANDALONE = "standalone"


class RolloutReplica(ABC):
    """Rollout replica is an individual server instance, which may be deployed on single or multiple nodes.
    It is equivalent to launch server in each node with command line:

    SGLang:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 2 ...
    python -m sglang.launch_server --node-rank 1 --nnode 2 ...
    ```

    vLLM:
    ```
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 0 ...
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 8 ...
    ```

    Args:
        replica_rank: int, rank of this rollout replica.
        config: RolloutConfig, full config.
        model_config: DictConfig, model config.
        gpus_per_node: int, number of gpus per node.
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ) -> None:
        self.replica_rank = replica_rank
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = model_config

        self.world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.gpus_per_node = gpus_per_node
        self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0, (
            f"world_size {self.world_size} must be divisible by gpus_per_node {self.gpus_per_replica_node}"
        )
        self.nnodes = self.world_size // self.gpus_per_replica_node
        self.is_reward_model = is_reward_model
        self.is_teacher_model = is_teacher_model
        self.name_suffix = f"_{name_suffix}" if name_suffix else ""

        self.rollout_mode: RolloutMode = None
        self.resource_pool: ResourcePool | None = None
        self.bundle_indices: list[int] = []

        self.servers: list[RemoteWorkerGroup] = []
        self._server_groups: list[WorkerGroup[Worker]] = []
        self._server_address: str = None
        self._server_handle: RemoteWorkerGroup | None = None
        self._worker_group: WorkerGroup[Worker] | None = None
        self._owns_worker_group = False

    def close(self) -> None:
        """Close WorkerGroup resources created by this replica.

        Process-level Runtime lifecycle stays with the composition that installed it.
        """
        from verl.runtime import ExceptionGroup

        errors: list[Exception] = []
        failed_server_groups: list[WorkerGroup[Worker]] = []
        for server_group in reversed(self._server_groups):
            try:
                server_group.close()
            except Exception as exc:  # noqa: BLE001 - close remaining sibling owners
                errors.append(exc)
                failed_server_groups.append(server_group)
        self._server_groups = list(reversed(failed_server_groups))

        if not self._server_groups:
            self.servers.clear()
            self._server_handle = None

        worker_group = self._worker_group
        if worker_group is not None and self._owns_worker_group:
            try:
                worker_group.close()
            except Exception as exc:  # noqa: BLE001 - report after closing peer owners
                errors.append(exc)
            else:
                self._worker_group = None
                self._owns_worker_group = False
        else:
            self._worker_group = None
            self._owns_worker_group = False

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("runtime failures", errors)

    @property
    def worker_group(self) -> WorkerGroup[Worker]:
        """Return the exact WorkerGroup view used by checkpoint synchronization."""
        if self._worker_group is None:
            raise RuntimeError("rollout replica WorkerGroup is not initialized")
        return self._worker_group

    async def init_hybrid(self, worker_group: WorkerGroup[Worker], resource_pool: ResourcePool):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: WorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.rollout_mode = RolloutMode.HYBRID
        start = self.world_size * self.replica_rank
        self._worker_group = worker_group.slice(start, self.world_size)
        self.resource_pool = resource_pool.slice(slice(start, start + self.world_size))
        self._owns_worker_group = False
        await self.launch_servers()

    async def init_hybrid_colocated(self, worker_group: WorkerGroup[Worker], resource_pool: ResourcePool):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: WorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
            resource_pool: RayResourcePool, ray placement group where hybrid engine processes have been launched.
            bundle_indices: list[int], bundle indices for this rollout replica.
        """
        self.rollout_mode = RolloutMode.HYBRID
        start = self.world_size * self.replica_rank
        self._worker_group = worker_group.slice(start, self.world_size)
        self.resource_pool = resource_pool.slice(slice(start, start + self.world_size))
        self._owns_worker_group = False
        self.bundle_indices = [self.replica_rank * self.world_size + idx for idx in range(self.world_size)]
        await self.launch_servers()

    # TODO(sgm): this should be the default solution, but need to make the RolloutMode more clear.
    async def init_colocated(self, resource_pool: ResourcePool):
        """Init colocated rollout server, rollout engine and hybrid engine colocated in same ray placement group
        but in separate processes.

        Args:
            resource_pool: RayResourcePool, ray placement group where hybrid engine processes have been launched.
        """
        self.rollout_mode = RolloutMode.COLOCATED
        self.resource_pool = resource_pool

        if self.is_reward_model:
            name_prefix = f"rollout_reward_colocate_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            name_prefix = f"rollout_teacher_colocate_{self.replica_rank}{self.name_suffix}"
        else:
            name_prefix = f"rollout_colocate_{self.replica_rank}{self.name_suffix}"

        worker_group = self._create_rollout_worker_group(name_prefix=name_prefix)
        self._worker_group = worker_group
        self._owns_worker_group = True
        await self.launch_servers()

    async def init_standalone(self, resource_pool: ResourcePool | None = None) -> None:
        """Init standalone rollout on a preselected per-replica pool.

        When ``resource_pool`` is omitted, preserve the legacy behavior and
        derive a fresh pool from the Runtime cluster.
        """
        self.rollout_mode = RolloutMode.STANDALONE
        if resource_pool is None:
            runtime = current_runtime()
            process_on_nodes = [self.gpus_per_replica_node] * self.nnodes
            resource_pool = runtime.create_resource_pool(
                nnodes=len(process_on_nodes),
                processes_per_node=process_on_nodes[0],
                device_type="gpu",
            )
        self.resource_pool = resource_pool

        # create worker group for this rollout
        if self.is_reward_model:
            name_prefix = f"rollout_reward_standalone_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            name_prefix = f"rollout_teacher_standalone_{self.replica_rank}{self.name_suffix}"
        else:
            name_prefix = f"rollout_standalone_{self.replica_rank}{self.name_suffix}"
        worker_group = self._create_rollout_worker_group(name_prefix=name_prefix)
        self._worker_group = worker_group
        self._owns_worker_group = True
        await self.launch_servers()

    def get_class_with_init_args(self) -> ClassWithInitArgs:
        """Deferred constructor for colocated and standalone CheckpointEngineWorker ranks."""
        legacy_method = type(self).get_ray_class_with_init_args
        if legacy_method is not RolloutReplica.get_ray_class_with_init_args:
            return legacy_method(self)
        from verl.checkpoint_engine.base import CheckpointEngineWorker

        return ClassWithInitArgs(
            CheckpointEngineWorker,
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
        )

    def get_ray_class_with_init_args(self) -> ClassWithInitArgs:
        """DEPRECATED: Use :meth:`get_class_with_init_args`."""
        return self.get_class_with_init_args()

    def _create_rollout_worker_group(self, *, name_prefix: str):
        """Create a Runtime-owned WorkerGroup on this replica's resource pool."""
        _ = name_prefix  # naming is owned by pool placement / Runtime actor naming
        return current_runtime().create_worker_group(self.get_class_with_init_args(), on=self.resource_pool)

    async def _create_server_worker_group(
        self,
        actor: ClassWithInitArgs[Worker],
        *,
        source_pool: ResourcePool,
    ) -> WorkerGroup[Worker]:
        runtime = current_runtime()
        sidecar_pool = runtime.create_resource_pool(
            nnodes=source_pool.nnodes,
            processes_per_node=1,
            device_type="cpu",
            on=source_pool,
        )
        wg = cast(
            WorkerGroup[Worker],
            await runtime.create_worker_group_async(
                actor,
                on=sidecar_pool,
            ),
        )
        self._server_groups.append(wg)
        return wg

    @staticmethod
    def _merge_cuda_visible_devices(worker_devices: list[str], *, expected_count: int) -> str:
        """Translate per-Worker physical visibility into one node-level list."""
        devices: list[str] = []
        for value in worker_devices:
            if not value or value == "not set":
                raise RuntimeError("GPU Worker did not report its visible devices")
            for device in value.split(","):
                device = device.strip()
                if device and device not in devices:
                    devices.append(device)
        if not devices:
            raise RuntimeError("GPU Workers reported an empty visible-device set")
        if len(devices) != expected_count:
            raise RuntimeError(f"GPU Workers reported {len(devices)} distinct devices; expected {expected_count}")
        return ",".join(devices)

    async def _set_server_endpoints(self, endpoints: list[RemoteWorkerGroup]) -> None:
        if self._worker_group is None:
            raise RuntimeError("rollout worker group is not initialized")
        if len(endpoints) != self._worker_group.world_size:
            raise ValueError(
                f"server endpoint count {len(endpoints)} does not match rollout world size "
                f"{self._worker_group.world_size}"
            )
        call = self._worker_group.submit("set_server_endpoint", args=(endpoints,))
        await call

    @abstractmethod
    async def launch_servers(self):
        """Launch http server in each node."""
        raise NotImplementedError

    @property
    def server_address(self) -> str:
        """Get rollout server address for OpenAI chat completion."""
        return self._server_address

    @property
    def server_handle(self) -> RemoteWorkerGroup:
        """Get rollout server handle for Token-in-token-out generation."""
        if self._server_handle is None:
            raise RuntimeError("rollout server is not initialized")
        return self._server_handle

    @property
    def max_concurrency(self) -> int:
        # 1000 is Ray's default max_concurrency for async execution.
        # Add some margin to account for control method call.
        return max(1000, self.config.max_num_seqs + CONTROL_METHOD_CONCURRENCY)

    def rollout_worker_use_gpu(self) -> bool:
        return True

    async def wake_up(self):
        """Wake up each rollout server."""
        await asyncio.gather(*[server.submit("wake_up") for server in self.servers])

    async def sleep(self):
        """Sleep each rollout server."""
        await asyncio.gather(*[server.submit("sleep") for server in self.servers])

    async def abort_all_requests(self):
        """Partial rollout: abort and save all unfinished requests in each rollout server."""
        await asyncio.gather(*[server.submit("abort_all_requests") for server in self.servers])

    async def resume_generation(self):
        """Resume generation on all servers after abort_all_requests."""
        await asyncio.gather(*[server.submit("resume_generation") for server in self.servers])

    async def clear_kv_cache(self):
        """reset kv cache in each rollout server."""
        await asyncio.gather(*[server.submit("clear_kv_cache") for server in self.servers])

    async def release_kv_cache(self):
        """Release only the kv_cache GPU memory, keeping model weights in place."""
        await asyncio.gather(*[server.submit("release_kv_cache") for server in self.servers])

    async def resume_kv_cache(self):
        """Restore the kv_cache GPU memory after a weight sync."""
        await asyncio.gather(*[server.submit("resume_kv_cache") for server in self.servers])

    async def start_profile(self, **kwargs):
        """Start profiling on the replica."""
        await asyncio.gather(*[server.submit("start_profile", kwargs=kwargs) for server in self.servers])

    async def stop_profile(self):
        """Stop profiling on the replica."""
        await asyncio.gather(*[server.submit("stop_profile") for server in self.servers])


class RolloutReplicaRegistry:
    """Factory for managing rollout replica implementations."""

    _registry: dict[str, Callable[[], type[RolloutReplica]]] = {}

    @classmethod
    def register(cls, name: str, loader: Callable[[], type[RolloutReplica]]) -> None:
        """Register a new rollout replica type."""
        cls._registry[name] = loader

    @classmethod
    def get(cls, name: str) -> type[RolloutReplica]:
        """Get a rollout replica class by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown rollout mode: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]()


# Loader functions for built-in types
def _load_vllm():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    return vLLMReplica


def _load_sglang():
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica

    return SGLangReplica


def _load_trtllm():
    from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica

    return TRTLLMReplica


# Register built-in types
RolloutReplicaRegistry.register("vllm", _load_vllm)
RolloutReplicaRegistry.register("sglang", _load_sglang)
RolloutReplicaRegistry.register("trtllm", _load_trtllm)


def get_rollout_replica_class(rollout: str, disaggregation_enabled: bool = False) -> type[RolloutReplica]:
    """Resolve a replica class by backend name.

    PD-disaggregated rollouts reuse the base backend name (``sglang`` /
    ``vllm``); the dispatch here picks the PD class only when the caller
    asserts ``disaggregation_enabled=True`` (sourced from
    ``RolloutConfig.disaggregation.enabled``). Validation in
    ``RolloutConfig.__post_init__`` rejects the flag for backends without a
    PD class.
    """
    if disaggregation_enabled:
        if rollout == "sglang":
            from verl.workers.rollout.sglang_rollout.sglang_pd_replica import SGLangPDReplica

            return SGLangPDReplica
        if rollout == "vllm":
            from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

            return vLLMPDReplica
        raise NotImplementedError(
            f"PD disaggregation is only supported with rollout in ('sglang', 'vllm'); got {rollout!r}."
        )
    return RolloutReplicaRegistry.get(rollout)
