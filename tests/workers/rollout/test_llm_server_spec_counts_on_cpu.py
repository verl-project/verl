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

import pytest

from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput


def _counts(drafts: int, accepted: int, verify_steps: int) -> dict:
    """A generation's speculative-decoding counts as the server reports them."""
    return {
        "spec_num_draft_tokens": drafts,
        "spec_num_accepted_tokens": accepted,
        "spec_num_verify_steps": verify_steps,
    }


def _generate(monkeypatch: pytest.MonkeyPatch, segments: list[TokenOutput]) -> TokenOutput:
    """The client's generation when each resume attempt returns the next segment (the base class is patched)."""
    pending = iter(segments)

    async def fake_generate(self, request_id, *, prompt_ids, **kwargs):
        return next(pending)

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake_generate)
    real_sleep = asyncio.sleep

    async def no_wait(_delay, *args, **kwargs):
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(llm_server.asyncio, "sleep", no_wait)
    client = FullyAsyncLLMServerClient(config=SimpleNamespace(), load_balancer_handle=None)
    return asyncio.run(client.generate(request_id="req-0", prompt_ids=[1, 2, 3], sampling_params={}))


def test_spec_counts_of_every_attempt_are_summed(monkeypatch):
    """An abort with counts, an abort before any token (the server reports none), then completion: the sums."""
    segments = [
        TokenOutput(token_ids=[101, 102], stop_reason="aborted", extra_fields={"global_steps": 5, **_counts(6, 4, 2)}),
        TokenOutput(token_ids=[], stop_reason="aborted", extra_fields={"global_steps": 5}),
        TokenOutput(
            token_ids=[103, 104, 105], stop_reason="completed", extra_fields={"global_steps": 6, **_counts(9, 3, 3)}
        ),
    ]

    output = _generate(monkeypatch, segments)

    assert output.token_ids == [101, 102, 103, 104, 105]
    assert {key: output.extra_fields[key] for key in _counts(0, 0, 0)} == _counts(15, 7, 5)


def test_generation_without_a_drafter_carries_no_spec_counts(monkeypatch):
    """A server without speculative decoding reports no counts, and the client adds none."""
    segment = TokenOutput(token_ids=[101], stop_reason="completed", extra_fields={"global_steps": 5})

    output = _generate(monkeypatch, [segment])

    assert not any(key in output.extra_fields for key in _counts(0, 0, 0))
