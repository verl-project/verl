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
"""GPU test: the weight-version cache_salt reaches SGLang's prefix cache.

A standalone server answers one prompt repeatedly and SGLang reports how many prompt tokens it
served from the radix cache. The version is bumped without any cache flush, so a miss can only
come from the salt: "salt" misses, "reset" still hits. A real abort -> set_global_steps ->
resume sequence is then checked in "salt" mode.

Usage (one GPU):
  pytest tests/workers/rollout/rollout_sglang/test_sglang_prefix_cache_versioning.py -v -s

Environment overrides:
  PREFIX_CACHE_TEST_MODEL_PATH - model (default ~/models/Qwen/Qwen2.5-0.5B-Instruct)
"""

import asyncio
import os
import sys

import pytest
import ray
from hydra import compose, initialize_config_dir

from verl.workers.rollout.replica import get_rollout_replica_class

MODEL_PATH = os.path.expanduser(os.getenv("PREFIX_CACHE_TEST_MODEL_PATH", "~/models/Qwen/Qwen2.5-0.5B-Instruct"))
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../../../verl/trainer/config")

# Long enough to fill several KV pages.
PROMPT_IDS = list(range(1000, 1400))


def _make_config(mode):
    with initialize_config_dir(config_dir=os.path.abspath(CONFIG_DIR), version_base=None):
        config = compose(config_name="ppo_trainer")
    rollout = config.actor_rollout_ref.rollout
    config.actor_rollout_ref.model.path = MODEL_PATH
    rollout.name = "sglang"
    rollout.tensor_model_parallel_size = 1
    rollout.prompt_length = 512
    rollout.response_length = 16
    rollout.max_model_len = 1024
    rollout.max_num_seqs = 8
    rollout.gpu_memory_utilization = 0.4
    rollout.load_format = "auto"
    rollout.enable_prefix_caching = True
    rollout.external_kv_cache_invalidation = mode
    return config


def _record_cached_tokens(server):
    # verl's SGLang server does not surface cached_tokens, so record it from the engine output.
    tokenizer_manager = server.tokenizer_manager
    original = tokenizer_manager.generate_request
    server.recorded_cached_tokens = []

    async def generate_request(obj, request=None):
        async for output in original(obj, request):
            server.recorded_cached_tokens.append(output["meta_info"].get("cached_tokens", 0))
            yield output

    tokenizer_manager.generate_request = generate_request


def _cached_tokens(handle, request_id):
    ray.get(
        handle.generate.remote(
            prompt_ids=PROMPT_IDS,
            sampling_params={"temperature": 0.0, "max_tokens": 4},
            request_id=request_id,
        )
    )
    return ray.get(handle.__ray_call__.remote(lambda server: server.recorded_cached_tokens[-1]))


@pytest.mark.parametrize("mode", ["reset", "salt"])
def test_salt_isolates_sglang_prefix_cache(mode):
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "false"}})
    try:
        config = _make_config(mode)
        replica = get_rollout_replica_class("sglang")(
            replica_rank=0,
            config=config.actor_rollout_ref.rollout,
            model_config=config.actor_rollout_ref.model,
            gpus_per_node=1,
        )
        asyncio.run(replica.init_standalone())
        handle = replica._server_handle
        # Ship the recorder by value: the server's worker cannot import this test module.
        ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
        ray.get(handle.__ray_call__.remote(_record_cached_tokens))

        ray.get(handle.set_global_steps.remote(1))
        assert _cached_tokens(handle, "v1-first") == 0
        assert _cached_tokens(handle, "v1-again") > 0, "same version must hit"

        ray.get(handle.set_global_steps.remote(2))
        hit = _cached_tokens(handle, "v2-no-flush")
        if mode == "salt":
            assert hit == 0, "a new weight version must not reuse the previous version's prefix"
        else:
            assert hit > 0, "without a flush, 'reset' mode keeps serving the old prefix"

        if mode == "salt":
            asyncio.run(replica.abort_all_requests())
            ray.get(handle.set_global_steps.remote(3))
            asyncio.run(replica.resume_generation())
            assert _cached_tokens(handle, "v3-first") == 0
            assert _cached_tokens(handle, "v3-again") > 0
    finally:
        ray.shutdown()
