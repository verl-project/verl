# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer import main_generation_server
from verl.utils.profiler.nsight import _build_rollout_ray_actor_nsight_launch_options


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [None, False, True], ids=["unconfigured", "disabled", "whole-process"])
async def test_start_server_prepares_a_private_rollout_profiler_config(monkeypatch, enabled):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"name": "vllm", "tensor_model_parallel_size": 1, "profiler": None},
                "model": {},
            },
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
            "global_profiler": {
                "global_tool_config": {
                    "nsys": {
                        "worker_nsight_options": {"capture-range": "none", "trace": "cuda", "o": "standalone_{rank}_%p"}
                    }
                }
            },
        }
    )
    if enabled is not None:
        config.actor_rollout_ref.rollout.profiler = {
            "tool": "nsys",
            "enable": enabled,
            "ranks": [0],
            "tool_config": {"nsys": {"discrete": False}},
        }
    original = OmegaConf.to_container(config, resolve=True)
    OmegaConf.set_struct(config, True)
    replica = SimpleNamespace(init_standalone=AsyncMock(), _server_handle=object(), _server_address="test-server")
    replica_class = MagicMock(return_value=replica)
    monkeypatch.setattr(main_generation_server, "get_rollout_replica_class", lambda name: replica_class)

    await main_generation_server.start_server(config)

    received = replica_class.call_args.kwargs["config"]
    options = _build_rollout_ray_actor_nsight_launch_options(
        received.profiler, replica_rank=0, replica_world_size=1, role="rollout_server_node_0"
    )
    if enabled:
        assert options["capture-range"] == "none"
        assert options["trace"] == "cuda"
        # A replica may adjust its private launch options without changing the caller's config.
        received.profiler.ray_actor_nsight_options["o"] = "replica-local-output"
    else:
        assert options is None
    assert received is not config.actor_rollout_ref.rollout
    assert OmegaConf.to_container(config, resolve=True) == original
