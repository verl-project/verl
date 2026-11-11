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

"""Client restart preserves tokens, log probabilities, routes, top-k heads, and response budgets."""

import copy

import numpy as np
import pytest
from omegaconf import OmegaConf

from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, _partial_rollout_session
from verl.workers.rollout.replica import TokenOutput


def _client(response_length: int = 4) -> FullyAsyncLLMServerClient:
    return FullyAsyncLLMServerClient(
        config=OmegaConf.create(
            {"actor_rollout_ref": {"rollout": {"response_length": response_length, "name": "vllm"}}}
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("first_reason", ["aborted", "completed"])
async def test_restore_prefix_and_completed_session(monkeypatch, first_reason):
    calls = []
    states: dict[str, TokenOutput] = {}

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        calls.append((list(prompt_ids), dict(sampling_params)))
        version = len(calls)
        return TokenOutput(
            token_ids=[10 * version, 10 * version + 1],
            log_probs=[-float(version), -float(version)],
            stop_reason=first_reason if version == 1 else "length",
            extra_fields={
                "global_steps": version,
                "response_topk_ids": np.array([[version, version + 1], [version + 2, version + 3]], dtype=np.int32),
                "response_topk_log_probs": np.array([[-0.1, -0.9], [-0.2, -0.8]], dtype=np.float32),
            },
        )

    class Restart(Exception):
        pass

    async def interrupt_sleep(_delay):
        raise Restart

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    monkeypatch.setattr(llm_server.asyncio, "sleep", interrupt_sleep)
    token = _partial_rollout_session.set((states, "0", None))
    try:
        if first_reason == "aborted":
            with pytest.raises(Restart):
                await _client().generate("old", prompt_ids=[1, 2], sampling_params={"temperature": 1})
        else:
            await _client().generate("old", prompt_ids=[1, 2], sampling_params={"temperature": 1})
    finally:
        _partial_rollout_session.reset(token)

    state = states["0"].model_dump()
    retained = copy.deepcopy(state)
    output = await _client().generate(
        "new",
        prompt_ids=[1, 2],
        sampling_params={"temperature": 1},
        partial_rollout_state=state,
    )
    if first_reason == "completed":
        assert len(calls) == 1
        assert output.token_ids == [10, 11]
        assert output.extra_fields["min_global_steps"] == 1
        assert output.extra_fields["max_global_steps"] == 1
    else:
        assert calls[1][0] == [1, 2, 10, 11]
        assert calls[1][1]["max_tokens"] == 2
        assert output.token_ids == [10, 11, 20, 21]
        assert output.extra_fields["min_global_steps"] == 1
        assert output.extra_fields["max_global_steps"] == 2
        assert output.extra_fields["response_topk_ids"].shape == (4, 2)
    # Resume must never modify the serialized checkpoint in place.
    assert state["token_ids"] == retained["token_ids"]
    assert state["extra_fields"].keys() == retained["extra_fields"].keys()


@pytest.mark.asyncio
async def test_exact_budget_prefix_finishes_without_backend_call(monkeypatch):
    async def fail_generate(self, *args, **kwargs):
        raise AssertionError("Backend should not be called when remaining budget is zero")

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fail_generate)
    state = TokenOutput(
        token_ids=[10, 11, 12, 13],
        log_probs=[-0.1, -0.2, -0.3, -0.4],
        stop_reason="aborted",
        extra_fields={"global_steps": 2, "min_global_steps": 2, "max_global_steps": 2},
    ).model_dump()
    output = await _client(response_length=4).generate(
        "new",
        prompt_ids=[1],
        sampling_params={"temperature": 1.0},
        partial_rollout_state=state,
    )
    assert output.token_ids == [10, 11, 12, 13]
    assert output.stop_reason == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "match"),
    [
        ({"token_ids": [1, 2, 3, 4, 5], "stop_reason": "aborted"}, "exceeds response budget"),
        ({"token_ids": "not-a-token-list"}, "Input should be a valid list"),
    ],
)
async def test_incompatible_resume_fails_before_backend(monkeypatch, state, match):
    async def fail_generate(self, *args, **kwargs):
        raise AssertionError("Backend should not be called on invalid checkpoint state")

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fail_generate)
    with pytest.raises(ValueError, match=match):
        await _client(response_length=4).generate(
            "new",
            prompt_ids=[1],
            sampling_params={"temperature": 1.0},
            partial_rollout_state=state,
        )
