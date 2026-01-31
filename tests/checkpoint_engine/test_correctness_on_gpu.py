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
import os

import pytest
import ray

from tests.checkpoint_engine.test_utils import create_rollout_worker_group, create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray.base import (
    RayResourcePool,
    split_resource_pool,
)
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("rebuild_group", [False, True])
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 6)])
async def test_nccl_checkpoint_engine(
    rebuild_group,
    num_trainer,
    num_rollout,
    num_nodes=1,
    num_gpus_per_node=8,
    check_allclose=True,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    model_path = os.path.expanduser(model_path)
    ray.init(
        runtime_env={
            "env_vars": {
                "UCX_TLS": "rc,tcp,cuda",
                "UCX_MAX_RNDV_RAILS": "4",
                "UCX_LOG_LEVEL": "INFO",
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    # initialize config
    checkpoint_engine_config = CheckpointEngineConfig(
        backend="nccl", engine_kwargs={"nccl": {"rebuild_group": rebuild_group}}
    )
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(name="vllm", checkpoint_engine=checkpoint_engine_config)

    # create trainer and rollout worker group
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    trainer = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    trainer.reset()
    rollout, replicas = await create_rollout_worker_group(rollout_pool, model_config, rollout_config, check_allclose)

    # create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(backend="nccl", trainer=trainer, replicas=replicas)
    for _ in range(3):
        await checkpoint_manager.update_weights()
        rollout.check_weights()

    ray.shutdown()


@pytest.mark.skip(reason="temporary skip since our ci environment is not ready")
@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 6)])
async def test_nixl_checkpoint_engine(
    num_trainer,
    num_rollout,
    device,
    num_nodes=1,
    num_gpus_per_node=8,
    check_allclose=True,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    model_path = os.path.expanduser(model_path)
    ray.init(
        runtime_env={
            "env_vars": {
                # TODO: it's pretty hard to set these environment variables right, please consult
                # with your network admin. Maybe auto adjust UCX_* according to NCCL_IB_*?
                "UCX_TLS": "rc,ud,cuda",
                # "UCX_IB_GID_INDEX": "3", # NCCL_IB_GID_INDEX
                # "UCX_IB_DEVICES": "mlx5_1:1,mlx5_2:1,mlx5_3:1", # NCCL_IB_HCA
                "UCX_RC_TIMEOUT": "30s",  # NCCL_IB_TIMEOUT
                "UCX_RC_RETRY_COUNT": "7",  # NCCL_IB_RETRY_COUNT
                "UCX_KEEPALIVE_INTERVAL": "1s",
                "UCX_KEEPALIVE_NUM_EPS": "10",
                "UCX_MAX_RNDV_RAILS": "4",
                "UCX_IB_ROCE_REACHABILITY_MODE": "all",
                "UCX_LOG_LEVEL": "INFO",
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    # initialize config
    checkpoint_engine_config = CheckpointEngineConfig(backend="nixl", engine_kwargs={"nixl": {"device": device}})
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(name="vllm", checkpoint_engine=checkpoint_engine_config)

    # create trainer and rollout worker group
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    trainer = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    trainer.reset()
    rollout, replicas = await create_rollout_worker_group(rollout_pool, model_config, rollout_config, check_allclose)

    # create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(backend="nixl", trainer=trainer, replicas=replicas)
    for _ in range(3):
        await checkpoint_manager.update_weights()
        rollout.check_weights()

    ray.shutdown()


if __name__ == "__main__":
    test_nccl_checkpoint_engine(
        rebuild_group=False,
        num_trainer=2,
        num_rollout=30,
        num_nodes=4,
        num_gpus_per_node=8,
        check_allclose=False,
        model_path=os.environ["HDFS_ROOT"] + "/model/Qwen3-30B-A3B-Base",
    )
