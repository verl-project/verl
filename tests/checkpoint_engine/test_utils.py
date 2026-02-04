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
from typing import Generator

import ray
import torch
from transformers import AutoModelForCausalLM

from verl.checkpoint_engine import CheckpointEngineRegistry, CheckpointEngineWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.workers.config import CheckpointEngineConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout import BaseRollout, RolloutReplica


class TrainingWorkerTest(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig, checkpoint_engine_config: CheckpointEngineConfig) -> None:
        super().__init__(config)
        backend = checkpoint_engine_config.backend
        bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
        engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        per_tensor_param, _ = self.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)


class MockServerAdapter(BaseRollout):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, check_allclose: bool = True):
        super().__init__(config, model_config, device_mesh=None)
        self.check_allclose = check_allclose
        self.model = None
        self.received_weights: dict[str, torch.Tensor] = {}

    async def resume(self, tags: list[str]):
        raise NotImplementedError()

    async def release(self):
        raise NotImplementedError()

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        async for name, weight in weights:
            weight = weight.clone()
            if self.check_allclose:
                self.received_weights[name] = weight.clone()

    def check_weights(self):
        if not self.check_allclose:
            return

        if self.model is None:
            local_path = copy_to_local(self.model_config.path)
            self.model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16, device_map="cpu")

        for name, weight in self.model.state_dict().items():
            assert name in self.received_weights, f"weight {name} not received"
            received = self.received_weights[name]
            assert torch.allclose(weight.to(received.device), received), f"weight {name} not equal"
        self.received_weights.clear()


class MockReplica(RolloutReplica):
    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        raise NotImplementedError

    async def launch_servers(self):
        """Launch http server in each node."""
        raise NotImplementedError


class CheckpointEngineWorkerTest(CheckpointEngineWorker):
    def __init__(self, rollout_config: RolloutConfig, model_config: HFModelConfig, check_allclose: bool = True) -> None:
        server_adapter = MockServerAdapter(rollout_config, model_config, check_allclose)
        super().__init__(rollout_config, model_config, server_adapter)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_weights(self):
        self.server_adapter.check_weights()


def create_trainer_worker_group(
    resource_pool: RayResourcePool, model_config: HFModelConfig, checkpoint_engine_config: CheckpointEngineConfig
) -> RayWorkerGroup:
    engine_config = FSDPEngineConfig(forward_only=True, fsdp_size=resource_pool.world_size, strategy="fsdp")
    trainer_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
    )

    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(TrainingWorkerTest),
        config=trainer_config,
        checkpoint_engine_config=checkpoint_engine_config,
    )
    ray_cls_with_init.update_options(
        {
            "runtime_env": {
                "env_vars": {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            }
        }
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name())
    return wg


async def create_rollout_worker_group(
    resource_pool: RayResourcePool,
    model_config: HFModelConfig,
    rollout_config: RolloutConfig,
    check_allclose: bool = True,
) -> tuple[RayWorkerGroup, list[MockReplica]]:
    # create rollout worker group
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(CheckpointEngineWorkerTest),
        model_config=model_config,
        rollout_config=rollout_config,
        check_allclose=check_allclose,
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name())

    # create rollout replicas
    rollout_world_size = (
        rollout_config.tensor_model_parallel_size
        * rollout_config.data_parallel_size
        * rollout_config.pipeline_model_parallel_size
    )
    num_replicas = wg.world_size // rollout_world_size
    replicas = []
    for replica_rank in range(num_replicas):
        replica = MockReplica(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
        )
        replicas.append(replica)
    await asyncio.gather(*[replica.init_hybrid(wg) for replica in replicas])

    return wg, replicas
