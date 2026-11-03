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
"""A WeightUpdateAgent keeps a real vLLM replica on the newest published weights, under load.

The test process publishes three versions: the original weights, then weights whose final norm is
zeroed (so every logit is 0 and greedy decoding degenerates), then the original weights again.
Greedy outputs show which weights the replica serves, and every output reports its weight version.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_weight_update_agent.py -v -s
"""

import asyncio
import os
from uuid import uuid4

import pytest
import ray
import torch

MODEL_PATH = os.path.expanduser(os.getenv("WEIGHT_UPDATE_AGENT_TEST_MODEL", "~/models/Qwen/Qwen2.5-0.5B-Instruct"))
GREEDY = {"temperature": 0.0, "top_p": 1.0, "logprobs": False}


def _rollout_config():
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")
    config.actor_rollout_ref.model.path = MODEL_PATH
    rollout = config.actor_rollout_ref.rollout
    rollout.name = "vllm"
    rollout.mode = "async"
    rollout.tensor_model_parallel_size = 1
    rollout.prompt_length = 256
    rollout.response_length = 512
    rollout.gpu_memory_utilization = 0.5
    rollout.load_format = "auto"
    rollout.checkpoint_engine.backend = "transport_pull"
    rollout.checkpoint_engine.update_weights_bucket_megabytes = 256
    return config.actor_rollout_ref


def _versions():
    """Original weights, weights with the final norm zeroed, original weights again."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    original = [(name, tensor.detach().clone()) for name, tensor in model.state_dict().items()]
    broken = [(name, torch.zeros_like(t) if name == "model.norm.weight" else t) for name, t in original]
    assert any(name == "model.norm.weight" for name, _ in original)
    return [original, broken, original]


@pytest.mark.skipif(not os.path.isdir(MODEL_PATH), reason=f"{MODEL_PATH} not found")
def test_agent_updates_vllm_replica_under_load():
    pytest.importorskip("nixl")
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"VLLM_USE_V1": "1"}})
    try:
        asyncio.run(_run())
    finally:
        ray.shutdown()


async def _run():
    from transformers import AutoTokenizer

    from verl.checkpoint_engine.transport import RayWeightRegistry
    from verl.checkpoint_engine.transport.publisher import WeightPublisher
    from verl.checkpoint_engine.transport.update_agent import WeightUpdateAgent
    from verl.workers.rollout.replica import get_rollout_replica_class

    actor_rollout_ref = _rollout_config()
    replica = get_rollout_replica_class("vllm")(
        replica_rank=0, config=actor_rollout_ref.rollout, model_config=actor_rollout_ref.model, gpus_per_node=1
    )
    await replica.init_standalone()
    server = replica.servers[0]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Name three primary colors."}], add_generation_prompt=True, tokenize=True
        )
    )

    async def generate(max_tokens):
        return await server.generate.remote(
            prompt_ids=prompt_ids, sampling_params={**GREEDY, "max_tokens": max_tokens}, request_id=uuid4().hex
        )

    # The replica loaded the checkpoint from disk; this is what the original weights produce.
    reference = (await generate(16)).token_ids

    # The registry and engine defaults match those of the replica's transport_pull workers.
    publisher = WeightPublisher(RayWeightRegistry(), bucket_size=256 << 20, device="cpu", backend="nixl_native")
    agent = ray.remote(WeightUpdateAgent).remote(servers=replica.servers, workers=replica.workers, poll_interval=0.1)
    run = agent.run.remote()
    versions = _versions()

    # Version 0: the original weights.
    await publisher.publish(iter(versions[0]), version=0)
    assert await agent.wait_for_version.remote(0, 300) == 0
    output = await generate(16)
    assert output.token_ids == reference
    assert output.extra_fields["global_steps"] == 0

    # Version 1, published while a long request is generating: the apply phase must abort it.
    long_request = asyncio.ensure_future(generate(512))
    await asyncio.sleep(0.5)
    await publisher.publish(iter(versions[1]), version=1)
    assert await agent.wait_for_version.remote(1, 300) == 1
    interrupted = await long_request
    assert interrupted.stop_reason in ("aborted", "abort"), interrupted.stop_reason
    output = await generate(16)
    assert output.token_ids != reference, "the replica still serves the original weights"
    assert len(set(output.token_ids)) == 1, "with all-zero logits greedy decoding repeats one token"
    assert output.extra_fields["global_steps"] == 1

    # Version 2: the original weights again.
    await publisher.publish(iter(versions[2]), version=2)
    assert await agent.wait_for_version.remote(2, 300) == 2
    output = await generate(16)
    assert output.token_ids == reference
    assert output.extra_fields["global_steps"] == 2

    state = await agent.get_state.remote()
    assert not state["failed"] and state["weight_update/pause_seconds"] > 0
    await agent.stop.remote()
    await run
