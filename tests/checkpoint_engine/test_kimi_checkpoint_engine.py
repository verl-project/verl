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
from verl.single_controller.ray.base import (
    RayResourcePool,
    split_resource_pool,
)
from verl.utils.device import get_device_name


@pytest.mark.parametrize("rebuild_group", [False, True])
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 6)])
def test_kimi_checkpoint_engine(
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

    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
    resource_pool.get_placement_groups(device_name=get_device_name())
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    checkpoint_kwargs = {
        "train_world_size": num_trainer,
        "rollout_world_size": num_rollout,
        "bucket_size": 2 * 1024 * 1024 * 1024,  # 2GB
        "rebuild_group": rebuild_group,
    }

    trainer = create_trainer_worker_group(model_path, trainer_pool, "kimi_ckpt_engine", checkpoint_kwargs)
    trainer.reset()
    rollout = create_rollout_worker_group(
        model_path, rollout_pool, "kimi_ckpt_engine", checkpoint_kwargs, check_allclose=check_allclose
    )

    world_size = trainer.world_size + rollout.world_size
    for _ in range(3):
        # 1. prepare all workers
        metadata = ray.get(
            trainer.execute_checkpoint_engine(["prepare"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        )
        trainer_kwargs = {
            "method": ["init_process_group"] * trainer.world_size,
            "rank": list(range(0, trainer.world_size)),
            "world_size": [world_size] * trainer.world_size,
            "master_metadata": [metadata[0]] * trainer.world_size,
        }
        rollout_kwargs = {
            "method": ["init_process_group"] * rollout.world_size,
            "rank": list(range(trainer.world_size, world_size)),
            "world_size": [world_size] * rollout.world_size,
            "master_metadata": [metadata[0]] * rollout.world_size,
        }

        # 2. init process group between all workers
        ray.get(
            trainer.execute_checkpoint_engine(**trainer_kwargs) + rollout.execute_checkpoint_engine(**rollout_kwargs)
        )

        # 3. update weights of all workers
        ray.get(trainer.update_weights() + rollout.update_weights())

        # 4. finish all workers
        ray.get(
            trainer.execute_checkpoint_engine(["finish"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["finish"] * rollout.world_size)
        )

        # 5. check weights of rollout workers
        rollout.check_weights()

    ray.shutdown()


if __name__ == "__main__":
    test_kimi_checkpoint_engine(
        rebuild_group=False,
        num_trainer=2,
        num_rollout=30,
        num_nodes=2,
        num_gpus_per_node=16,
        check_allclose=False,
        model_path=os.environ["HDFS_ROOT"] + "/model/Qwen3-30B-A3B-Base",
    )
