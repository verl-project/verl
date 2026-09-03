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

import pytest

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.protocol import DataProto


class _FakeRemoteComputeScore:
    def __init__(self):
        self.received_data: DataProto | None = None

    async def remote(self, data: DataProto) -> dict:
        self.received_data = data
        return {"reward_score": 1.0, "reward_extra_info": {}}


class _FakeRewardLoopWorkerHandle:
    def __init__(self):
        self.compute_score = _FakeRemoteComputeScore()


class _DummyAgentLoopWorker:
    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _get_mm_processor_kwargs = AgentLoopWorker._get_mm_processor_kwargs
    _compute_score = AgentLoopWorker._compute_score

    def __init__(self, reward_loop_worker_handle: _FakeRewardLoopWorkerHandle):
        self.processor = None
        self.mm_processor_kwargs = {}
        self.reward_loop_worker_handles = [reward_loop_worker_handle]


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
async def test_async_reward_data_proto_preserves_validate_meta_info(validate: bool):
    reward_loop_worker_handle = _FakeRewardLoopWorkerHandle()
    worker = _DummyAgentLoopWorker(reward_loop_worker_handle)
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
    )

    await worker._compute_score([output], kwargs={}, validate=validate)

    received_data = reward_loop_worker_handle.compute_score.received_data
    assert received_data is not None
    assert received_data.meta_info == {"validate": validate}
