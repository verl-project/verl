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
from verl.utils.device import get_device_name
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("rebuild_group", [False])
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 6)])
async def test_hccl_checkpoint_engine(
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
                "HCCL_CONNECT_TIMEOUT": "1500",
                "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
                "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    # initialize config
    checkpoint_engine_config = CheckpointEngineConfig(
        backend="hccl", engine_kwargs={"hccl": {"rebuild_group": rebuild_group}}
    )
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(name="vllm", checkpoint_engine=checkpoint_engine_config)

    # create trainer and rollout worker group
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
    resource_pool.get_placement_groups(device_name=get_device_name())
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    trainer = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    trainer.reset()
    rollout, replicas = await create_rollout_worker_group(rollout_pool, model_config, rollout_config, check_allclose)

    # create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(backend="hccl", trainer=trainer, replicas=replicas)
    for _ in range(3):
        await checkpoint_manager.update_weights()
        rollout.check_weights()

    ray.shutdown()


if __name__ == "__main__":
    test_hccl_checkpoint_engine(
        rebuild_group=False,
        num_trainer=2,
        num_rollout=6,
        num_nodes=1,
        num_gpus_per_node=8,
        check_allclose=False,
        model_path=os.environ["HDFS_ROOT"] + "/model/Qwen3-30B-A3B-Base",
    )
