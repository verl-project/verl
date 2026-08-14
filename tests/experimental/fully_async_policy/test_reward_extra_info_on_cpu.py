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

"""Regression coverage for fully-async reward metadata finalization."""

import concurrent.futures

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncAgentLoopManager
from verl.protocol import DataProto


class _FakeObjectRef:
    def __init__(self, output):
        self.output = output

    def future(self):
        future = concurrent.futures.Future()
        future.set_result(self.output)
        return future


class _FakeRemoteMethod:
    def __init__(self, output):
        self.output = output

    def remote(self, prompts):
        assert isinstance(prompts, DataProto)
        return _FakeObjectRef(self.output)


class _FakeWorker:
    def __init__(self, output):
        self.generate_sequences = _FakeRemoteMethod(output)


@pytest.mark.asyncio
async def test_fully_async_manager_finalizes_reward_extra_info():
    prompts = DataProto(
        batch=TensorDict({"input_ids": torch.ones((1, 1), dtype=torch.long)}, batch_size=1),
        non_tensor_batch={"uid": np.array(["sample-0"], dtype=object)},
    )
    worker_output = DataProto(
        batch=TensorDict({"rm_scores": torch.ones((1, 1), dtype=torch.float32)}, batch_size=1),
        non_tensor_batch={"reward_extra_info": np.array([{"score": 1.0, "acc": True, "pred": "42"}], dtype=object)},
        meta_info={"metrics": [{}]},
    )
    manager = object.__new__(FullyAsyncAgentLoopManager)
    manager.agent_loop_workers = [_FakeWorker(worker_output)]

    generate_sequences_single = FullyAsyncAgentLoopManager.generate_sequences_single
    while hasattr(generate_sequences_single, "__wrapped__"):
        generate_sequences_single = generate_sequences_single.__wrapped__
    result = await generate_sequences_single(manager, prompts)

    assert result.meta_info["reward_extra_keys"] == ["score", "acc", "pred"]
    assert "reward_extra_info" not in result.non_tensor_batch
    assert result.non_tensor_batch["score"].tolist() == [1.0]
    assert result.non_tensor_batch["acc"].tolist() == [True]
    assert result.non_tensor_batch["pred"].tolist() == ["42"]
