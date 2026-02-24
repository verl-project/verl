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
import os

import pytest
import ray
from omegaconf import DictConfig
from openai import AsyncOpenAI, OpenAI

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.checkpoint_engine import CheckpointEngineManager
from verl.workers.rollout.replica import get_rollout_replica_class


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 2
    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    config.actor_rollout_ref.model.path = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.skip_tokenizer_init = False

    return config


@pytest.mark.asyncio
@pytest.mark.parametrize("tp_size", [2, 4])
async def test_standalone_rollout(init_config, tp_size):
    """Test standalone rollout single node and multi nodes."""
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "NCCL_P2P_DISABLE": "1",  # disable p2p in L20
            }
        }
    )

    init_config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp_size
    num_replicas = (init_config.trainer.n_gpus_per_node * init_config.trainer.nnodes) // tp_size
    rollout_config = init_config.actor_rollout_ref.rollout
    model_config = init_config.actor_rollout_ref.model

    # create standalone rollout server
    rollout_server_class = get_rollout_replica_class(init_config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank, config=rollout_config, model_config=model_config, gpus_per_node=2
        )
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("NO_PROXY", None)

    client = AsyncOpenAI(
        api_key="123-abc",
        base_url=f"http://{server_addresses[0]}/v1",
    )

    completion = await client.chat.completions.create(
        model=init_config.actor_rollout_ref.model.path,
        messages=[{"role": "user", "content": "What can you do?"}],
    )
    print(completion.choices[0].message.content)

    ray.shutdown()


@pytest.mark.skip(reason="local test only")
def test_hybrid_rollout_with_ep(init_config):
    """Test hybrid rollout with expert parallelism, DP=2, TP=4, EP=8."""
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    model_path = os.path.expanduser("~/models/Qwen/Qwen3-30B-A3B-Instruct-2507")
    init_config.actor_rollout_ref.model.path = model_path

    # parallelism config
    init_config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
    init_config.actor_rollout_ref.rollout.data_parallel_size = 4
    init_config.actor_rollout_ref.rollout.expert_parallel_size = 8

    # 1. init hybrid worker: FSDP+rollout
    # - build FSDP model and optimizer
    # - offload FSDP model and optimizer, build rollout
    # - sleep rollout and load FSDP model and optimizer
    agent_loop_manager = init_agent_loop_manager(init_config)
    checkpoint_manager = CheckpointEngineManager(
        backend=init_config.actor_rollout_ref.rollout.checkpoint_engine.backend,
        trainer=agent_loop_manager.worker_group,
        replicas=agent_loop_manager.rollout_replicas,
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()

    # 3. test async openai call
    server_address = agent_loop_manager.server_addresses[0]
    client = OpenAI(
        api_key="123-abc",
        base_url=f"http://{server_address}/v1",
    )

    smapling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
    }

    response = client.chat.completions.create(
        model=model_path,
        messages=[{"role": "user", "content": "What can you do?"}],
        **smapling_params,
    )

    completion = response.choices[0].message.content
    print(f"response: {completion}")

    print("Test passed!")
    ray.shutdown()
