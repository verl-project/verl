# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from openai import AsyncOpenAI

from verl.workers.rollout.replica import get_rollout_replica_class


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    # Inter-node settings: 2 GPUs split as 2 "nodes" x 1 GPU
    config.trainer.n_gpus_per_node = 1
    config.trainer.nnodes = 2
    model_root = os.path.expanduser(os.getenv("TRTLLM_TEST_MODEL_PATH_ROOT", "~/models"))
    config.actor_rollout_ref.model.path = os.path.join(model_root, "Qwen/Qwen2.5-0.5B-Instruct")
    config.actor_rollout_ref.rollout.name = "trtllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.skip_tokenizer_init = False

    return config


@pytest.mark.asyncio
async def test_inter_node_trtllm_rollout(init_config):
    """Test TRT-LLM rollout with TP=2 spanning 2 simulated nodes (1 GPU each).

    On CI with 2+ GPUs on 1 node, we set gpus_per_node=1 so init_standalone()
    creates resource_pool_spec=[1, 1] (2 placement groups), exercising the
    inter-node code path in get_pgs_and_bundle_indices() and launch_servers().
    """
    tp_size = 2  # fixed: only valid TP for Qwen2.5-0.5B-Instruct (2 KV heads)
    ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

    try:
        init_config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp_size
        num_replicas = (init_config.trainer.n_gpus_per_node * init_config.trainer.nnodes) // tp_size
        rollout_config = init_config.actor_rollout_ref.rollout
        model_config = init_config.actor_rollout_ref.model

        rollout_server_class = get_rollout_replica_class("trtllm")
        rollout_servers = [
            rollout_server_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=init_config.trainer.n_gpus_per_node,
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

    finally:
        ray.shutdown()
