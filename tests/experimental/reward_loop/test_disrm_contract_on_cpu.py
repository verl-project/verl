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

from types import SimpleNamespace

import pytest

from verl.experimental.reward_loop.reward_loop import RewardLoopWorker


@pytest.mark.parametrize(
    ("engine_name", "response", "expected_score", "expected_endpoint"),
    [
        ("vllm", {"data": [{"probs": [0.125, 0.875]}]}, 0.875, "classify"),
        ("sglang", {"data": [{"embedding": [0.25, 0.75]}]}, 0.75, "v1/embeddings"),
    ],
)
def test_compute_score_disrm_returns_reward_manager_contract(engine_name, response, expected_score, expected_endpoint):
    import asyncio

    asyncio.run(
        _test_compute_score_disrm_returns_reward_manager_contract(
            engine_name, response, expected_score, expected_endpoint
        )
    )


async def _test_compute_score_disrm_returns_reward_manager_contract(
    engine_name, response, expected_score, expected_endpoint
):
    worker = object.__new__(RewardLoopWorker)
    worker.config = SimpleNamespace(
        reward=SimpleNamespace(
            reward_model=SimpleNamespace(
                rollout=SimpleNamespace(name=engine_name),
                model_path="dummy-reward-model",
            )
        )
    )
    calls = []

    async def preprocess(data):
        return "reward prompt"

    async def post_request(payload, endpoint):
        calls.append((payload, endpoint))
        return response

    worker._preprocess_reward_inputs = preprocess
    worker._post_request = post_request

    result = await worker.compute_score_disrm(data=object())

    assert result == {"reward_score": expected_score, "reward_extra_info": {}}
    assert len(calls) == 1
    assert calls[0][1] == expected_endpoint
