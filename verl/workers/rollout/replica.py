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
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional

from omegaconf import DictConfig
from pydantic import BaseModel
from ray.actor import ActorHandle

from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, ResourcePoolManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import is_torch_npu_available
from verl.workers.config import HFModelConfig, RolloutConfig

logger = logging.getLogger(__file__)


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
    ) -> None:
        self.replica_rank = replica_rank
        self.config = omega_conf_to_dataclass(config)
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

        self.rollout_mode: RolloutMode = None
        self.workers: list[ActorHandle] = []
        self.resource_pool: RayResourcePool = None
        self.bundle_indices: list[int] = []

        self.servers: list[ActorHandle] = []
        self._server_address: str = None
        self._server_handle: ActorHandle = None

    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]
        await self.launch_servers()

    async def init_hybrid_colocated(self, worker_group: RayWorkerGroup, resource_pool: RayResourcePool):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
            resource_pool: RayResourcePool, ray placement group where hybrid engine processes have been launched.
            bundle_indices: list[int], bundle indices for this rollout replica.
        """
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]
        self.resource_pool = resource_pool
        self.bundle_indices = [self.replica_rank * self.world_size + idx for idx in range(self.world_size)]
        await self.launch_servers()

    # TODO(sgm): this should be the default solution, but need to make the RolloutMode more clear.
    async def init_colocated(self, resource_pool: RayResourcePool):
        """Init colocated rollout server, rollout engine and hybrid engine colocated in same ray placement group
        but in separate processes.

        Args:
            resource_pool: RayResourcePool, ray placement group where hybrid engine processes have been launched.
        """
        self.rollout_mode = RolloutMode.COLOCATED
        self.resource_pool = resource_pool
        use_gpu = self.rollout_worker_use_gpu()

        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"rollout_colocate_{self.replica_rank}"
            if not self.is_reward_model
            else f"rollout_reward_colocate_{self.replica_rank}",
            use_gpu=use_gpu,
            device_name="cuda" if not is_torch_npu_available(check_device=False) else "npu",
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    async def init_standalone(self):
        """Init standalone rollout server, create new resource pool for this rollout."""
        # create resource pool for this rollout
        self.rollout_mode = RolloutMode.STANDALONE
        resource_pool_name = (
            f"rollout_pool_{self.replica_rank}"
            if not self.is_reward_model
            else f"rollout_pool_reward_{self.replica_rank}"
        )
        resource_pool_spec = {
            resource_pool_name: [self.gpus_per_replica_node] * self.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=None)
        resource_pool_manager.create_resource_pool()
        self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]

        # create worker group for this rollout
        use_gpu = self.rollout_worker_use_gpu()
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"rollout_standalone_{self.replica_rank}"
            if not self.is_reward_model
            else f"rollout_reward_standalone_{self.replica_rank}",
            use_gpu=use_gpu,
            device_name="cuda" if not is_torch_npu_available(check_device=False) else "npu",
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    @abstractmethod
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        raise NotImplementedError

    @abstractmethod
    async def launch_servers(self):
        """Launch http server in each node."""
        raise NotImplementedError

    @property
    def server_address(self) -> str:
        """Get rollout server address for OpenAI chat completion."""
        return self._server_address

    @property
    def server_handle(self) -> ActorHandle:
        """Get rollout server handle for Token-in-token-out generation."""
        return self._server_handle

    def rollout_worker_use_gpu(self) -> bool:
        return True

    async def wake_up(self):
        """Wake up each rollout server."""
        await asyncio.gather(*[server.wake_up.remote() for server in self.servers])

    async def sleep(self):
        """Sleep each rollout server."""
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self):
        """Partial rollout: abort and save all unfinished requests in each rollout server."""
        # TODO(wuxibin)
        # await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])
        print(f"abort all requests in rollout replica {self.replica_rank}")

    async def resume_all_requests(self):
        """Partial rollout: resume all unfinished requests in each rollout server."""
        # TODO(wuxibin)
        # await asyncio.gather(*[server.resume_all_requests.remote() for server in self.servers])
        print(f"resume all requests in rollout replica {self.replica_rank}")

    async def clear_kv_cache(self):
        """reset kv cache in each rollout server."""
        await asyncio.gather(*[server.clear_kv_cache.remote() for server in self.servers])

    async def start_profile(self, **kwargs):
        """Start profiling on the replica."""
        await asyncio.gather(*[server.start_profile.remote(**kwargs) for server in self.servers])

    async def stop_profile(self):
        """Stop profiling on the replica."""
        await asyncio.gather(*[server.stop_profile.remote() for server in self.servers])


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
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

    try:
        import vllm  # noqa: F401
    except ImportError:
        import sys
        import types
        from unittest.mock import Mock

        mock_vllm = types.ModuleType("vllm")

        mock_custom_ops = types.ModuleType("vllm._custom_ops")
        mock_custom_ops.scaled_fp8_quant = Mock()
        mock_vllm._custom_ops = mock_custom_ops

        mock_model_executor = types.ModuleType("vllm.model_executor")
        mock_layers = types.ModuleType("vllm.model_executor.layers")
        mock_activation = types.ModuleType("vllm.model_executor.layers.activation")

        class GeluAndMul:  # noqa: N801
            pass

        class SiluAndMul:  # noqa: N801
            pass

        mock_activation.GeluAndMul = GeluAndMul
        mock_activation.SiluAndMul = SiluAndMul
        mock_layers.activation = mock_activation
        mock_model_executor.layers = mock_layers
        mock_vllm.model_executor = mock_model_executor

        sys.modules["vllm"] = mock_vllm
        sys.modules["vllm._custom_ops"] = mock_custom_ops
        sys.modules["vllm.model_executor"] = mock_model_executor
        sys.modules["vllm.model_executor.layers"] = mock_layers
        sys.modules["vllm.model_executor.layers.activation"] = mock_activation

    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica

    del os.environ["SGLANG_USE_CPU_ENGINE"]
    return SGLangReplica


def _load_trtllm():
    from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica

    return TRTLLMReplica


# Register built-in types
RolloutReplicaRegistry.register("vllm", _load_vllm)
RolloutReplicaRegistry.register("sglang", _load_sglang)
RolloutReplicaRegistry.register("trtllm", _load_trtllm)


# Original function for backward compatibility
def get_rollout_replica_class(rollout: str) -> type[RolloutReplica]:
    return RolloutReplicaRegistry.get(rollout)
