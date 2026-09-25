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
"""Regression tests for the standalone DisRM result contract (issue #7368).

``RewardLoopWorker.compute_score_disrm`` used to return only ``{"reward_score": ...}``.
The async consumer ``AgentLoopWorker._compute_score`` reads ``result["reward_extra_info"]``
unconditionally, so standalone DisRM scoring raised ``KeyError: 'reward_extra_info'`` for
every trajectory and no training batch was produced. These tests pin the two-field
contract at the producer and end-to-end through the real consumer indexing.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from verl.experimental.reward_loop.reward_loop import RewardLoopWorker


def _make_worker(engine_name: str) -> RewardLoopWorker:
    worker = object.__new__(RewardLoopWorker)
    worker.config = SimpleNamespace(
        reward=SimpleNamespace(
            reward_model=SimpleNamespace(
                rollout=SimpleNamespace(name=engine_name),
                model_path="dummy-reward-model",
            )
        )
    )
    worker._preprocess_reward_inputs = AsyncMock(return_value="reward prompt")
    return worker


@pytest.mark.parametrize(
    ("engine_name", "response", "expected_score", "expected_endpoint"),
    [
        ("vllm", {"data": [{"probs": [0.125, 0.875]}]}, 0.875, "classify"),
        ("sglang", {"data": [{"embedding": [0.25, 0.75]}]}, 0.75, "v1/embeddings"),
    ],
)
def test_compute_score_disrm_returns_two_field_contract(engine_name, response, expected_score, expected_endpoint):
    worker = _make_worker(engine_name)
    worker._post_request = AsyncMock(return_value=response)

    result = asyncio.run(worker.compute_score_disrm(data=object()))

    # Both fields required by the reward-manager contract must be present.
    assert result == {"reward_score": expected_score, "reward_extra_info": {}}
    worker._post_request.assert_awaited_once()
    assert worker._post_request.await_args.args[1] == expected_endpoint


def test_disrm_result_survives_agent_loop_consumer_indexing():
    """End-to-end: the real consumer that used to KeyError now completes scoring."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker

    class _RemoteMethod:
        def __init__(self, worker):
            self.worker = worker

        def remote(self, data):
            return self.worker.compute_score_disrm(data)

    class _RewardWorkerHandle:
        def __init__(self, worker):
            self.compute_score = _RemoteMethod(worker)

    async def _run():
        producer = _make_worker("vllm")
        producer._post_request = AsyncMock(return_value={"data": [{"probs": [0.125, 0.875]}]})

        consumer = object.__new__(AgentLoopWorker)
        consumer.reward_loop_worker_handles = [_RewardWorkerHandle(producer)]
        consumer._compute_multi_modal_inputs = lambda output, input_ids: {}
        consumer._compute_position_ids = lambda input_ids, attention_mask, mm, mm_processor_kwargs=None: torch.arange(
            input_ids.shape[-1], dtype=torch.long
        ).unsqueeze(0)
        consumer._get_mm_processor_kwargs = lambda audio_data=None: {}

        output = AgentLoopOutput(
            prompt_ids=[1, 2],
            response_ids=[3],
            response_mask=[1],
            metrics=AgentLoopMetrics(),
            extra_fields={},
        )
        # Must not raise KeyError('reward_extra_info').
        await consumer._compute_score([output], kwargs={})
        assert output.reward_score == pytest.approx(0.875)
        assert output.extra_fields["reward_extra_info"] == {}

    asyncio.run(_run())
