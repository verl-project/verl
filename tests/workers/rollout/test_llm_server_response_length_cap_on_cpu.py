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
"""Cumulative response budget across partial rollout resumes.

A resume re-sends ``prompt + everything generated so far`` as ``prompt_ids``. If the request
carries no explicit limit, the server falls back to

    min(response_length, prompt_length + response_length - len(prompt_ids))

which charges already-generated tokens against the *prompt* budget and measures that budget with
the configured ``prompt_length`` rather than this request's actual prompt length. Generation then
runs to ``prompt_length + response_length - actual_prompt_length`` -- the overshoot is exactly the
unused prompt budget, is decoded at the deepest KV positions, and is discarded by the agent loop's
``[: response_length]`` slice. The ``max_model_len`` clamp does not bound it, because
``max_model_len`` defaults to the model's ``max_position_embeddings``.

``FakeRolloutServer`` below mirrors ``vLLMHttpServer.generate``'s cap arithmetic so these tests
pin the behaviour the real servers implement.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput

PROMPT_LENGTH = 2048
RESPONSE_LENGTH = 8192
MAX_MODEL_LEN = 10241


class FakeRolloutServer:
    """Stand-in for the rollout server, reproducing its ``max_tokens`` resolution.

    The model never emits EOS, so a request only ends when it exhausts the budget it was granted;
    every ``abort_every`` tokens a weight sync aborts it, which is what drives the resume loop.
    """

    def __init__(self, abort_every: int):
        self.abort_every = abort_every
        self.grants: list[tuple[int, int]] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        # Requests cross a Ray boundary, so the server only ever mutates its own copy.
        params = dict(sampling_params)

        max_possible_tokens = MAX_MODEL_LEN - len(prompt_ids)
        assert max_possible_tokens >= 1

        if "max_tokens" in params:
            max_tokens = params.pop("max_tokens")
        elif "max_new_tokens" in params:
            max_tokens = params.pop("max_new_tokens")
        else:
            max_tokens = min(RESPONSE_LENGTH, PROMPT_LENGTH + RESPONSE_LENGTH - len(prompt_ids))
        max_tokens = max(1, min(max_tokens, max_possible_tokens))
        self.grants.append((len(prompt_ids), max_tokens))

        generated = min(max_tokens, self.abort_every)
        return TokenOutput(
            token_ids=[7] * generated,
            log_probs=None,
            stop_reason="length" if generated == max_tokens else "aborted",
        )


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeRolloutServer:
    fake = FakeRolloutServer(abort_every=2000)
    # ``super().generate`` binds to the class, so the base class has to be patched. An already
    # bound method is not a descriptor, so it does not pick up the client as a second ``self``.
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake.generate)

    real_sleep = asyncio.sleep

    async def no_wait(_delay, *args, **kwargs):
        return await real_sleep(0, *args, **kwargs)

    # The resume loop backs off 1s between attempts; collapse it to keep the test fast.
    monkeypatch.setattr(llm_server.asyncio, "sleep", no_wait)
    return fake


def _config(*, include_async_training: bool = True) -> Any:
    config: dict[str, Any] = {"actor_rollout_ref": {"rollout": {"response_length": RESPONSE_LENGTH, "name": "vllm"}}}
    if include_async_training:
        config["async_training"] = {"partial_rollout": True}
    return OmegaConf.create(config)


async def _generate(config: Any, prompt_len: int, sampling_params: dict[str, Any]) -> TokenOutput:
    client = FullyAsyncLLMServerClient(config=config, load_balancer_handle=None)
    return await client.generate(
        request_id="req-0",
        prompt_ids=list(range(prompt_len)),
        sampling_params=sampling_params,
    )


# A prompt far shorter than ``prompt_length`` leaves the most unused budget to leak into the
# response; a prompt that fills ``prompt_length`` leaves none, which is why the pre-fix overshoot
# went unnoticed in configs whose prompts are near the limit.
@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_len", [92, 136, PROMPT_LENGTH])
async def test_resumes_never_exceed_response_length(server, prompt_len):
    output = await _generate(_config(), prompt_len, {"temperature": 1.0})

    assert len(output.token_ids) <= RESPONSE_LENGTH, (
        f"generated {len(output.token_ids)} tokens against a {RESPONSE_LENGTH} budget; "
        f"per-attempt grants were {server.grants}"
    )
    # A model that never emits EOS spends the budget exactly, and each attempt is granted only
    # what the response still owes, so the grants sum to the budget.
    assert len(output.token_ids) == RESPONSE_LENGTH
    assert server.grants[0][1] == RESPONSE_LENGTH
    assert sum(min(grant, server.abort_every) for _, grant in server.grants) == RESPONSE_LENGTH
    assert output.stop_reason == "length"


@pytest.mark.asyncio
async def test_cap_holds_without_async_training_key(server):
    """v1 async trainers install this client with a config that has no ``async_training`` key, so
    ``should_retry`` stays True and the resume loop runs with no way to opt out."""
    output = await _generate(_config(include_async_training=False), 136, {"temperature": 1.0})

    assert len(output.token_ids) <= RESPONSE_LENGTH


@pytest.mark.asyncio
async def test_caller_sampling_params_not_mutated(server):
    """The per-attempt budget is written back into ``sampling_params``; the caller reuses that dict
    across turns, so the rewrite must land on a copy."""
    sampling_params = {"temperature": 1.0, "top_p": 0.9}

    await _generate(_config(), 136, sampling_params)

    assert sampling_params == {"temperature": 1.0, "top_p": 0.9}


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_key", ["max_tokens", "max_new_tokens"])
async def test_explicit_caller_budget_wins(server, limit_key):
    output = await _generate(_config(), 136, {"temperature": 1.0, limit_key: 500})

    assert len(output.token_ids) <= 500


@pytest.mark.asyncio
async def test_config_without_rollout_section_is_tolerated(server):
    """Lightweight callers and tests pass a config stub; the client must fall back to the server
    default rather than raising."""
    output = await _generate(SimpleNamespace(), 136, {"temperature": 1.0})

    assert len(output.token_ids) > 0
