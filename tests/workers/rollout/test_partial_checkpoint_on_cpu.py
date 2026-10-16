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

"""Client restart preserves tokens, log probabilities, routes, and response budgets."""

import copy

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput


def _client():
    return FullyAsyncLLMServerClient(
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"response_length": 4, "name": "vllm"}}})
    )


def _serialized(state):
    state = copy.deepcopy(state)
    state["output"] = state["output"].model_dump()
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("first_reason", ["aborted", "completed"])
async def test_restore_prefix_and_completed_session(monkeypatch, first_reason):
    calls = []
    saved = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        calls.append(list(prompt_ids))
        version = len(calls)
        return TokenOutput(
            token_ids=[10 * version, 10 * version + 1],
            log_probs=[-version, -version],
            stop_reason=first_reason if version == 1 else "length",
            extra_fields={
                "global_steps": version,
            },
        )

    class Restart(Exception):
        pass

    def checkpoint(state):
        if state["output"].token_ids:
            saved.append(_serialized(state))
            raise Restart

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    with pytest.raises(Restart):
        await _client().generate(
            "old",
            prompt_ids=[1, 2],
            sampling_params={"temperature": 1},
            partial_rollout_checkpoint_callback=checkpoint,
        )
    state = saved[0]
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
    else:
        assert calls[1] == [1, 2, 10, 11]
        assert output.token_ids == [10, 11, 20, 21]
    # Resume must never modify the serialized checkpoint in place.
    assert state["output"]["token_ids"] == retained["output"]["token_ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["prompt", "sampling", "schema"])
async def test_incompatible_resume_fails_before_backend(monkeypatch, changed):
    states = []

    def checkpoint(state):
        states.append(_serialized(state))
        raise RuntimeError("snapshot")

    with pytest.raises(RuntimeError, match="snapshot"):
        await _client().generate(
            "old",
            prompt_ids=[1],
            sampling_params={"temperature": 1},
            partial_rollout_checkpoint_callback=checkpoint,
        )
    state = states[0]
    prompt, params = [1], {"temperature": 1}
    if changed == "prompt":
        prompt = [2]
    elif changed == "sampling":
        params["temperature"] = 0.5
    else:
        state["schema_version"] = 999
    with pytest.raises(ValueError, match="checkpoint"):
        await _client().generate("new", prompt_ids=prompt, sampling_params=params, partial_rollout_state=state)
