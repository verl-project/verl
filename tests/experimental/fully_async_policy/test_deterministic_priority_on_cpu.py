# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.fully_async_policy.detach_utils import prepare_single_generation_data
from verl.workers.rollout import llm_server as llm_server_module
from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer
from verl.workers.rollout.replica import TokenOutput


def _config(rollout_n: int):
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                n=rollout_n,
                multi_turn=SimpleNamespace(enable=False),
            )
        )
    )


def _batch(priority: int | None = None):
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    if priority is not None:
        batch["priority"] = np.array([priority], dtype=np.int64)
    return batch


def _priorities(sample_step: int, rollout_n: int = 4) -> np.ndarray:
    batch = prepare_single_generation_data(_batch(), _config(rollout_n), sample_step=sample_step)
    return batch.non_tensor_batch["priority"]


def test_expanded_rollouts_get_unique_int64_priorities():
    priorities = _priorities(sample_step=7, rollout_n=4)

    assert priorities.dtype == np.int64
    np.testing.assert_array_equal(priorities, np.array([28, 29, 30, 31], dtype=np.int64))


def test_consecutive_sample_steps_get_disjoint_priority_ranges():
    first = _priorities(sample_step=7)
    second = _priorities(sample_step=8)

    assert set(first).isdisjoint(second)
    assert first[-1] + 1 == second[0]


def test_existing_sample_priority_is_not_overwritten():
    batch = prepare_single_generation_data(_batch(priority=23), _config(4), sample_step=7)

    np.testing.assert_array_equal(batch.non_tensor_batch["priority"], np.array([23, 23, 23, 23], dtype=np.int64))


@pytest.mark.asyncio
async def test_expanded_priorities_reach_single_turn_as_distinct_request_ids():
    request_ids = []

    class CapturingServer:
        async def generate(self, **kwargs):
            request_ids.append(kwargs["request_id"])
            return TokenOutput(token_ids=[4], extra_fields={})

    agent_loop = object.__new__(SingleTurnAgentLoop)
    agent_loop.rollout_config = SimpleNamespace(full_determinism=True)
    agent_loop.response_length = 8
    agent_loop.server_manager = CapturingServer()

    async def process_multi_modal_info(_messages):
        return {}

    async def build_initial_tokens(_messages, **_kwargs):
        return [1, 2, 3]

    async def merge_assistant_token(prompt_ids, token_ids, *_args, **_kwargs):
        return SimpleNamespace(token_ids=[*prompt_ids, *token_ids]), [1] * len(token_ids), None

    agent_loop.process_multi_modal_info = process_multi_modal_info
    agent_loop._get_mm_processor_kwargs = lambda _audios: None
    agent_loop._assert_mm_supported = lambda _has_multimodal_data: None
    agent_loop.ct_build_initial_tokens = build_initial_tokens
    agent_loop.ct_merge_assistant_token = merge_assistant_token

    priorities = _priorities(sample_step=7)
    for priority in priorities:
        await agent_loop.run(sampling_params={}, priority=priority, raw_prompt=[])

    assert request_ids == [f"det-{priority}" for priority in priorities]
    assert len(set(request_ids)) == len(priorities)


def test_unique_request_ids_avoid_sticky_routing_collapse(monkeypatch):
    # Make the routing assertion independent of the process's PYTHONHASHSEED.
    monkeypatch.setattr(
        llm_server_module,
        "hash",
        lambda request_id: int(request_id.removeprefix("det-")),
        raising=False,
    )
    servers = {f"replica-{index}": None for index in range(4)}

    collapsed_lb = GlobalRequestLoadBalancer(servers, full_determinism=True)
    collapsed = [collapsed_lb.acquire_server("det-0")[0] for _ in range(16)]
    assert len(set(collapsed)) == 1

    priorities = np.concatenate([_priorities(sample_step=step) for step in range(1, 5)])
    distributed_lb = GlobalRequestLoadBalancer(servers, full_determinism=True)
    distributed = [distributed_lb.acquire_server(f"det-{priority}")[0] for priority in priorities]

    assert set(distributed) == set(servers)
    assert all(distributed.count(server_id) == 4 for server_id in servers)
