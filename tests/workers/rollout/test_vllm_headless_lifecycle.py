# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Only the replica's API node owns an AsyncLLM lifecycle client."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm")

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


@pytest.mark.parametrize("method", ["wake_up", "sleep", "clear_kv_cache"])
@pytest.mark.parametrize("node_rank", [0, 1])
def test_replica_lifecycle_broadcast_only_calls_api_engine(method, node_rank):
    server = SimpleNamespace(
        node_rank=node_rank,
        rollout_mode=RolloutMode.COLOCATED,
        config=SimpleNamespace(free_cache_engine=True),
        _get_wake_up_tags=lambda: ["weights"],
    )
    if node_rank == 0:
        server.engine = SimpleNamespace(
            **{
                name: AsyncMock()
                for name in ("wake_up", "sleep", "reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache")
            }
        )
    asyncio.run(getattr(vLLMHttpServer, method)(server))
    if node_rank == 0:
        called = {"wake_up": "wake_up", "sleep": "sleep", "clear_kv_cache": "reset_prefix_cache"}[method]
        getattr(server.engine, called).assert_awaited_once()
