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
from verl.workers.rollout.utils import ended_with_native_eos

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


def _input_processor(primary_eos=151643, extra_eos=151645):
    """Build a minimal vLLM input_processor stub with primary and generation_config EOS IDs."""
    renderer = SimpleNamespace(get_eos_token_id=lambda: primary_eos) if primary_eos is not None else None
    gen_fields = {"eos_token_id": extra_eos} if extra_eos is not None else {}
    return SimpleNamespace(renderer=renderer, generation_config_fields=gen_fields)


@pytest.mark.parametrize(
    "completion, ignore_eos, expected",
    [
        (SimpleNamespace(token_ids=[10, 20, 151643], finish_reason="stop"), False, True),
        # Secondary EOS from generation_config.json (e.g. Qwen3 <|im_end|> != tokenizer.eos_token_id).
        (SimpleNamespace(token_ids=[10, 20, 151645], finish_reason="stop"), False, True),
        # Native EOS emitted on the exact final max_tokens / max_model_len budget slot.
        (SimpleNamespace(token_ids=[10, 20, 151645], finish_reason="length"), False, True),
        # Hit max_tokens / max_model_len without emitting EOS.
        (SimpleNamespace(token_ids=[10, 20, 999], finish_reason="length"), False, False),
        # Stopped on a user stop string or non-EOS stop_token_id.
        (SimpleNamespace(token_ids=[10, 20, 999], finish_reason="stop"), False, False),
        # Aborted mid-generation or before decoding any output.
        (SimpleNamespace(token_ids=[10, 20, 151645], finish_reason="abort"), False, False),
        (None, False, False),
        (SimpleNamespace(token_ids=[], finish_reason="length"), False, False),
        # ignore_eos disables EOS termination semantics.
        (SimpleNamespace(token_ids=[10, 20, 151645], finish_reason="length"), True, False),
    ],
)
def test_ended_with_native_eos_cases(completion, ignore_eos, expected):
    """Verify native-EOS detection across primary/secondary EOS IDs, length caps, stop strings, and aborts."""
    assert ended_with_native_eos(completion, _input_processor(), ignore_eos=ignore_eos) is expected


def test_ended_with_native_eos_returns_none_without_engine_eos_ids():
    """Omit ended_with_eos (return None) when neither renderer nor generation_config exposes EOS IDs."""
    completion = SimpleNamespace(token_ids=[1, 2], finish_reason="stop")
    assert ended_with_native_eos(None, _input_processor(primary_eos=None, extra_eos=None)) is None
    assert ended_with_native_eos(completion, _input_processor(primary_eos=None, extra_eos=None)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("final_eos", [True, False])
async def test_resumes_preserve_final_segment_ended_with_eos(monkeypatch: pytest.MonkeyPatch, final_eos: bool):
    """Carry the final segment's native-EOS flag across partial-rollout resumes."""
    segments = iter(
        [
            TokenOutput(token_ids=[11, 12], stop_reason="aborted", extra_fields={"ended_with_eos": False}),
            TokenOutput(
                token_ids=[13, 151645 if final_eos else 99],
                stop_reason="completed",
                extra_fields={"ended_with_eos": final_eos},
            ),
        ]
    )

    async def fake_generate(self, **kwargs):
        return next(segments)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake_generate)
    monkeypatch.setattr(llm_server.asyncio, "sleep", no_sleep)

    output = await _generate(SimpleNamespace(), 2, {"max_tokens": 4})
    assert output.token_ids == [11, 12, 13, 151645 if final_eos else 99]
    assert output.extra_fields["ended_with_eos"] is final_eos


@pytest.mark.asyncio
async def test_abort_at_budget_clears_ended_with_eos(monkeypatch: pytest.MonkeyPatch):
    """Clear ended_with_eos if an aborted segment already fills original_max_tokens."""
    segments = iter([TokenOutput(token_ids=[11, 12], stop_reason="aborted", extra_fields={"ended_with_eos": True})])

    async def fake_generate(self, **kwargs):
        return next(segments)

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake_generate)

    output = await _generate(SimpleNamespace(), 2, {"max_tokens": 2})
    assert output.stop_reason == "length"
    assert output.extra_fields["ended_with_eos"] is False


@pytest.mark.asyncio
async def test_single_turn_postprocess_slice_clears_ended_with_eos():
    """Clear ended_with_eos when SingleTurnAgentLoop slices tokens beyond response_length."""
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

    loop = object.__new__(SingleTurnAgentLoop)
    loop.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 2, "full_determinism": False})
    loop.prompt_length = 4
    loop.response_length = 2
    loop.mm_processor_kwargs = {}
    loop._assert_mm_supported = lambda has_mm: None

    async def process_mm(_):
        return {}

    async def build_prompt(*args, **kwargs):
        return [1, 2]

    async def generate(**kwargs):
        return TokenOutput(token_ids=[10, 11, 151645], stop_reason="completed", extra_fields={"ended_with_eos": True})

    async def merge(prompt_ids, assistant_ids, response_mask, response_logprobs, assistant_logprobs=None):
        return SimpleNamespace(token_ids=list(prompt_ids) + list(assistant_ids)), [1] * len(assistant_ids), None

    loop.process_multi_modal_info = process_mm
    loop.ct_build_initial_tokens = build_prompt
    loop.server_manager = SimpleNamespace(generate=generate)
    loop.ct_merge_assistant_token = merge

    out = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "x"}])
    assert out.response_ids == [10, 11]
    assert out.extra_fields["ended_with_eos"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "turn_flags, cap, tool_exhausts_budget, expected",
    [
        # Tool call stopped on stop token (False), final answer ended with EOS (True).
        ([False, True], 8, False, True),
        # Tool call ended with EOS (True), final answer hit length cap without EOS (False).
        ([True, False], 8, False, False),
        # Final answer ended with EOS, but response_length slicing cut off the tail.
        ([False, True], 1, False, False),
        # Tool observation exhausted response_length before the follow-up generation ran.
        ([True], 2, True, False),
    ],
)
async def test_tool_agent_loop_refreshes_and_bounds_ended_with_eos(
    turn_flags: list[bool], cap: int, tool_exhausts_budget: bool, expected: bool
):
    """Refresh ended_with_eos across tool turns and clear it on tool-budget or slice truncation."""
    from unittest.mock import AsyncMock

    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.response_length = cap
    loop.max_assistant_turns = loop.max_user_turns = None
    loop.max_parallel_calls = 1
    loop.tools = {}
    loop.tool_schemas = []
    loop._assert_mm_supported = lambda _: None
    loop.process_multi_modal_info = AsyncMock(return_value={})
    loop._get_mm_processor_kwargs = lambda _: {}
    loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            side_effect=[
                TokenOutput(token_ids=[10 + i], extra_fields={"ended_with_eos": flag})
                for i, flag in enumerate(turn_flags)
            ]
        )
    )
    loop.tool_parser = SimpleNamespace(
        stop_token_ids=[],
        extract_tool_calls=AsyncMock(side_effect=[("call", [SimpleNamespace(tool_call_id="c0")]), ("done", [])]),
    )
    loop._build_assistant_message = lambda content, _: {"role": "assistant", "content": content}

    async def pending(data, sampling_params):
        data.prompt_ids = [1, 2]
        return AgentState.GENERATING

    async def merge_assistant(prompt_ids, token_ids, response_mask, response_logprobs, **kwargs):
        return SimpleNamespace(token_ids=prompt_ids + token_ids), response_mask + [1] * len(token_ids), None

    loop._handle_pending_state = pending
    loop.ct_merge_assistant_token = merge_assistant
    if tool_exhausts_budget:
        loop.ct_merge_context_msg = AsyncMock(
            return_value=(SimpleNamespace(token_ids=[1, 2, 10, 20, 21]), [1, 0, 0], None)
        )
        loop._call_tool = AsyncMock(return_value=(SimpleNamespace(text="obs", image=None, video=None), 0.0, {}))
    else:

        async def tools(data):
            return AgentState.GENERATING

        loop._handle_processing_tools_state = tools

    out = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}])
    assert out.extra_fields["ended_with_eos"] is expected
