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

import numpy as np
import pytest
import ray
from omegaconf import DictConfig

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name
from verl.utils.tokenizer import hf_tokenizer
from verl.workers.config import CheckpointEngineConfig
from verl.workers.engine_workers import ActorRolloutRefWorker


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    config.trainer.n_gpus_per_node = 8
    config.trainer.nnodes = 1
    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    config.actor_rollout_ref.model.path = os.path.expanduser("~/models/Qwen/Qwen3-VL-2B-Instruct")
    config.actor_rollout_ref.rollout.name = os.environ.get("ROLLOUT_NAME", "vllm")
    config.actor_rollout_ref.rollout.skip_tokenizer_init = False
    config.actor_rollout_ref.rollout.max_num_seqs = 256
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.8
    config.actor_rollout_ref.rollout.agent.num_workers = 2
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "naive"
    config.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes = 256
    config.actor_rollout_ref.rollout.enforce_eager = True

    return config


@pytest.mark.asyncio
def test_server_adapter_colocated_weight_update(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
                "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
            }
        }
    )

    # 0. init actor rollout worker group
    resource_pool = RayResourcePool(
        process_on_nodes=[init_config.trainer.n_gpus_per_node] * init_config.trainer.nnodes, max_colocate_count=3
    )
    actor_rollout_cls = ray.remote(ActorRolloutRefWorker)
    cls_dict = {
        "actor_rollout": RayClassWithInitArgs(
            cls=actor_rollout_cls, config=init_config.actor_rollout_ref, role="actor_rollout"
        )
    }
    ray_cls_with_init = create_colocated_worker_cls(cls_dict)
    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name()
    )
    spawn_wg = wg_dict.spawn(prefix_set=cls_dict.keys())
    actor_rollout_wg = spawn_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    # 1. create AgentLoopManager
    agent_loop_manager = AgentLoopManager(
        config=init_config,
        worker_group=actor_rollout_wg,
        rollout_resource_pool=resource_pool,
    )

    # 2. create CheckpointEngineManager
    checkpoint_engine_config: CheckpointEngineConfig = omega_conf_to_dataclass(
        init_config.actor_rollout_ref.rollout.checkpoint_engine
    )
    checkpoint_manager = CheckpointEngineManager(
        config=checkpoint_engine_config,
        trainer=actor_rollout_wg,
        replicas=agent_loop_manager.rollout_replicas,
    )
    checkpoint_manager.sleep_replicas()

    # 3. generate prompts
    raw_prompts = [
        [
            {
                "role": "user",
                "content": "This is a test for weight update. If the weight has been correctly "
                'updated and you understand my meaning, please respond with "Test Passed".',
            }
        ],
        [
            {
                "role": "user",
                "content": "This is a test for weight update. If the weight has been correctly "
                'updated and you understand my meaning, please respond with "Test Passed".',
            }
        ],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "agent_name": np.array(["single_turn_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    )

    # 4. update weights and generate sequences, check if the responses are correct
    for _ in range(3):
        checkpoint_manager.update_weights()
        result = agent_loop_manager.generate_sequences(batch)
        checkpoint_manager.sleep_replicas()

        # Check response
        tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
        responses = result.batch["responses"]
        response_mask = result.batch["response_mask"]

        for i in range(len(responses)):
            valid_tokens = responses[i][response_mask[i].bool()]
            response = tokenizer.decode(valid_tokens)
            assert "test passed" in response.lower(), f"Response does not contain 'test passed': {response}"

            print("=========================")
            print("[OUTPUT]:", response)
            print("---")

    ray.shutdown()
