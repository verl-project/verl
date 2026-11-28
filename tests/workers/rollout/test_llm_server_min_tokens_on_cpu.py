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
"""FullyAsyncLLMServerClient shrinks a caller-set minimum length with the budget on resume."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput


def _install_segments(monkeypatch: pytest.MonkeyPatch, segments: list[TokenOutput]) -> list[dict]:
    """Script the base client so each attempt returns the next segment; return each attempt's sampling params."""
    pending = iter(segments)
    seen: list[dict] = []

    async def fake_generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        seen.append(dict(sampling_params))
        return next(pending)

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake_generate)

    real_sleep = asyncio.sleep

    async def no_wait(_delay, *args, **kwargs):
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(llm_server.asyncio, "sleep", no_wait)
    return seen


def _client() -> FullyAsyncLLMServerClient:
    # A config without ``async_training`` leaves partial rollout enabled.
    return FullyAsyncLLMServerClient(config=SimpleNamespace(), load_balancer_handle=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("min_key, max_key", [("min_tokens", "max_tokens"), ("min_new_tokens", "max_new_tokens")])
async def test_resume_shrinks_min_tokens_with_budget(monkeypatch, min_key, max_key):
    seen = _install_segments(
        monkeypatch,
        [
            TokenOutput(token_ids=[101, 102, 103], stop_reason="aborted"),
            TokenOutput(token_ids=[104, 105], stop_reason="completed"),
        ],
    )

    output = await _client().generate(request_id="r", prompt_ids=[1, 2], sampling_params={min_key: 5, max_key: 6})

    assert output.token_ids == [101, 102, 103, 104, 105]
    assert seen[0][min_key] == 5 and seen[0][max_key] == 6
    assert seen[1][min_key] == 2 and seen[1][max_key] == 3


@pytest.mark.asyncio
async def test_resume_past_min_tokens_floors_at_zero(monkeypatch):
    seen = _install_segments(
        monkeypatch,
        [
            TokenOutput(token_ids=[101, 102, 103], stop_reason="aborted"),
            TokenOutput(token_ids=[104], stop_reason="completed"),
        ],
    )

    await _client().generate(request_id="r", prompt_ids=[1, 2], sampling_params={"min_tokens": 2, "max_tokens": 8})

    assert seen[1]["min_tokens"] == 0 and seen[1]["max_tokens"] == 5


@pytest.mark.asyncio
async def test_without_min_tokens_adds_none(monkeypatch):
    seen = _install_segments(
        monkeypatch,
        [
            TokenOutput(token_ids=[101], stop_reason="aborted"),
            TokenOutput(token_ids=[102], stop_reason="completed"),
        ],
    )

    await _client().generate(request_id="r", prompt_ids=[1, 2], sampling_params={"max_tokens": 4})

    assert "min_tokens" not in seen[1] and seen[1]["max_tokens"] == 3
