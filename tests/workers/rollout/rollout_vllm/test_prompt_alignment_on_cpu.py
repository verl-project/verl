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

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("vllm")

from verl.workers.config import RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


@pytest.mark.parametrize(
    "rollout_prompt, replay, raises",
    [
        ([1, 10, 10, 2], True, False),
        ([1, 10, 10, 10, 2], True, True),
        ([1, 10, 11, 2], True, True),
        ([1, 10, 10, 10, 2], False, False),
    ],
)
def test_generate_checks_expanded_replay_prompt(rollout_prompt, replay, raises):
    routes = np.zeros((4, 2, 2), dtype=np.uint16)
    submitted = []

    async def generate(**kwargs):
        submitted.append(kwargs["prompt"]["prompt_token_ids"])
        yield SimpleNamespace(
            prompt_token_ids=rollout_prompt,
            outputs=[SimpleNamespace(token_ids=[3], routed_experts=routes, finish_reason="stop")],
        )

    server = object.__new__(vLLMHttpServer)
    server.config = RolloutConfig(
        name="vllm", max_model_len=64, prompt_length=32, response_length=32, enable_rollout_routing_replay=replay
    )
    server.model_config = SimpleNamespace(
        lora_rank=0,
        lora={},
        processor=SimpleNamespace(
            image_processor=type("Glm5NextImageProcessor", (), {})(), image_token_id=10, video_token_id=11
        ),
    )
    server.engine = SimpleNamespace(generate=generate)
    server._disaggregation_role = None
    server._submission_paused = False
    server._admitting = 0
    server.replica_rank = 0
    server.global_steps = 1

    request = server.generate(prompt_ids=[1, 10, 10, 2], sampling_params={"max_tokens": 1}, request_id="test")
    if raises:
        with pytest.raises(ValueError, match="identical actor and rollout prompt tokens"):
            asyncio.run(request)
    else:
        result = asyncio.run(request)
        assert result.token_ids == [3]
        assert result.routed_experts is (routes if replay else None)
    assert submitted == [[1, 10, 2]]
    assert server._admitting == 0
