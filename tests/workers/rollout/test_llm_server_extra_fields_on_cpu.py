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
"""FullyAsyncLLMServerClient keeps the backend's own extra_fields across resume attempts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput


def _install_segments(monkeypatch: pytest.MonkeyPatch, segments: list[TokenOutput]) -> None:
    """Script the base client so each attempt returns the next segment."""
    pending = iter(segments)

    async def fake_generate(self, request_id, *, prompt_ids, **kwargs):
        return next(pending)

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake_generate)

    real_sleep = asyncio.sleep

    async def no_wait(_delay, *args, **kwargs):
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(llm_server.asyncio, "sleep", no_wait)


def _client() -> FullyAsyncLLMServerClient:
    # A config without ``async_training`` leaves partial rollout enabled.
    return FullyAsyncLLMServerClient(config=SimpleNamespace(), load_balancer_handle=None)


@pytest.mark.asyncio
async def test_single_attempt_keeps_backend_extra_fields(monkeypatch):
    payload = np.arange(6).reshape(2, 3)
    segment = TokenOutput(token_ids=[101, 102], stop_reason="completed")
    segment.extra_fields.update({"global_steps": 3, "multi_modal_output": payload, "text": "hello"})
    _install_segments(monkeypatch, [segment])

    output = await _client().generate(request_id="r", prompt_ids=[1, 2], sampling_params={})

    assert output.token_ids == [101, 102]
    assert output.extra_fields["text"] == "hello"
    np.testing.assert_array_equal(output.extra_fields["multi_modal_output"], payload)
    assert output.extra_fields["global_steps"] == 3


@pytest.mark.asyncio
async def test_single_attempt_keeps_spec_decode_counts(monkeypatch):
    segment = TokenOutput(token_ids=[101], stop_reason="completed")
    segment.extra_fields.update(
        {"spec_num_draft_tokens": 10, "spec_num_accepted_tokens": 6, "spec_num_verify_steps": 4}
    )
    _install_segments(monkeypatch, [segment])

    output = await _client().generate(request_id="r", prompt_ids=[1, 2], sampling_params={})

    assert output.extra_fields["spec_num_draft_tokens"] == 10
    assert output.extra_fields["spec_num_accepted_tokens"] == 6
    assert output.extra_fields["spec_num_verify_steps"] == 4


@pytest.mark.asyncio
async def test_resume_sums_spec_decode_counts(monkeypatch):
    first = TokenOutput(token_ids=[101], stop_reason="aborted")
    first.extra_fields.update({"spec_num_draft_tokens": 10, "spec_num_accepted_tokens": 6, "spec_num_verify_steps": 4})
    second = TokenOutput(token_ids=[102], stop_reason="completed")
    second.extra_fields.update({"spec_num_draft_tokens": 5, "spec_num_accepted_tokens": 3, "spec_num_verify_steps": 2})
    _install_segments(monkeypatch, [first, second])

    output = await _client().generate(request_id="r", prompt_ids=[1, 2], sampling_params={})

    assert output.extra_fields["spec_num_draft_tokens"] == 15
    assert output.extra_fields["spec_num_accepted_tokens"] == 9
    assert output.extra_fields["spec_num_verify_steps"] == 6


@pytest.mark.asyncio
async def test_resume_keeps_latest_extra_fields_and_merged_keys(monkeypatch):
    first = TokenOutput(token_ids=[101], stop_reason="aborted")
    first.extra_fields.update(
        {
            "global_steps": 3,
            "num_cached_tokens": 2,
            "response_topk_ids": np.array([[7]]),
            "response_topk_log_probs": np.array([[-0.1]]),
            "text": "first",
            "only_first": 1,
        }
    )
    second = TokenOutput(token_ids=[102, 103], stop_reason="completed")
    second.extra_fields.update(
        {
            "global_steps": 4,
            "num_cached_tokens": 0,
            "response_topk_ids": np.array([[8], [9]]),
            "response_topk_log_probs": np.array([[-0.2], [-0.3]]),
            "text": "second",
        }
    )
    _install_segments(monkeypatch, [first, second])

    output = await _client().generate(request_id="r", prompt_ids=[1, 2], sampling_params={})

    assert output.token_ids == [101, 102, 103]
    assert output.extra_fields["text"] == "second"
    assert output.extra_fields["only_first"] == 1
    np.testing.assert_array_equal(output.extra_fields["response_topk_ids"], [[7], [8], [9]])
    np.testing.assert_array_equal(output.extra_fields["response_topk_log_probs"], [[-0.1], [-0.2], [-0.3]])
    assert output.extra_fields["num_cached_tokens"] == 2
    assert output.extra_fields["min_global_steps"] == 3
    assert output.extra_fields["max_global_steps"] == 4
