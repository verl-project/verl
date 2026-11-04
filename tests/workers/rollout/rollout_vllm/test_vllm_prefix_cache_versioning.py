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
"""GPU test: external_kv_cache_invalidation on a vLLM server with an external KV store.

vLLM's native CPU offloading (``kv_offloading_size``) stands in for a shared store such as
MooncakeStoreConnector: both are KV connectors keyed by vLLM's prefix block hashes and both are
wiped by reset_prefix_cache(reset_connector=True). The server answers one prompt and reports
how many prompt tokens it did not recompute, from the local prefix cache or the connector.

Each weight update is the real abort -> set_global_steps -> resume sequence. In both modes the
first request after an update is cold. A second abort with no version change then clears the
local cache only in "salt" mode, so the prompt is served from the kept store, while
"reset" has wiped the store too.

Usage (one GPU):
  pytest tests/workers/rollout/rollout_vllm/test_vllm_prefix_cache_versioning.py -v -s

Environment overrides:
  PREFIX_CACHE_TEST_MODEL_PATH - model (default ~/models/Qwen/Qwen2.5-0.5B-Instruct)
"""

import asyncio
import os
import time

import pytest
import ray
from hydra import compose, initialize_config_dir

from verl.workers.rollout.replica import get_rollout_replica_class

MODEL_PATH = os.path.expanduser(os.getenv("PREFIX_CACHE_TEST_MODEL_PATH", "~/models/Qwen/Qwen2.5-0.5B-Instruct"))
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../../../verl/trainer/config")

# Long enough to fill several KV blocks.
PROMPT_IDS = list(range(1000, 1400))


def _make_config(mode):
    with initialize_config_dir(config_dir=os.path.abspath(CONFIG_DIR), version_base=None):
        config = compose(config_name="ppo_trainer")
    rollout = config.actor_rollout_ref.rollout
    config.actor_rollout_ref.model.path = MODEL_PATH
    rollout.name = "vllm"
    rollout.tensor_model_parallel_size = 1
    rollout.prompt_length = 512
    rollout.response_length = 16
    rollout.max_model_len = 1024
    rollout.max_num_seqs = 8
    rollout.gpu_memory_utilization = 0.4
    rollout.enforce_eager = True
    rollout.load_format = "auto"
    rollout.enable_prefix_caching = True
    rollout.external_kv_cache_invalidation = mode
    rollout.engine_kwargs.vllm = {"kv_offloading_size": 1}
    return config


def _cached_tokens(handle, request_id):
    output = ray.get(
        handle.generate.remote(
            prompt_ids=PROMPT_IDS,
            sampling_params={"temperature": 0.0, "max_tokens": 4},
            request_id=request_id,
        )
    )
    # Let the connector finish storing the request's blocks before the next step.
    time.sleep(1)
    return output.extra_fields["num_cached_tokens"]


def _weight_sync(replica, handle, global_steps):
    asyncio.run(replica.abort_all_requests())
    if global_steps is not None:
        ray.get(handle.set_global_steps.remote(global_steps))
    asyncio.run(replica.resume_generation())


@pytest.mark.parametrize("mode", ["reset", "salt"])
def test_external_kv_cache_invalidation_with_cpu_offloading_store(mode):
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "false"}})
    try:
        config = _make_config(mode)
        replica = get_rollout_replica_class("vllm")(
            replica_rank=0,
            config=config.actor_rollout_ref.rollout,
            model_config=config.actor_rollout_ref.model,
            gpus_per_node=1,
        )
        asyncio.run(replica.init_standalone())
        handle = replica._server_handle

        _weight_sync(replica, handle, 1)
        assert _cached_tokens(handle, "v1-first") == 0
        assert _cached_tokens(handle, "v1-again") > 0, "same version must hit"

        _weight_sync(replica, handle, 2)
        assert _cached_tokens(handle, "v2-first") == 0, "a new weight version must start cold"

        # Abort without a version change: the local cache is cleared in both modes.
        _weight_sync(replica, handle, None)
        hit = _cached_tokens(handle, "v2-after-local-reset")
        if mode == "salt":
            assert hit > 0, "salt mode must keep the external store and hit it"
        else:
            assert hit == 0, "reset mode wipes the external store"
    finally:
        ray.shutdown()
