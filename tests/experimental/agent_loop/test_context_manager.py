# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from typing import Any

import pytest

from verl.experimental.agent_loop.context_manager import (
    ContextState,
    SlidingWindowContextManager,
    SummarizerContextManager,
)


class _FakeTokenizer:
    """Char-level tokenizer mock for deterministic encode/decode in unit tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _build_state(
    *,
    prompt_text: str,
    response_text: str,
    messages: list[dict[str, Any]],
    response_mask: list[int] | None = None,
    response_logprobs: list[float] | None = None,
    routed_experts=None,
) -> ContextState:
    tokenizer = _FakeTokenizer()
    prompt_ids = tokenizer.encode(prompt_text)
    response_ids = tokenizer.encode(response_text)
    if response_mask is None:
        response_mask = [1] * len(response_ids)
    if response_logprobs is None:
        response_logprobs = []
    return ContextState(
        messages=messages,
        trajectory_ids=prompt_ids + response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        routed_experts=routed_experts,
        multi_modal_data={"images": ["keep-me"]},
        reward_score=1.0,
        num_turns=3,
        extra_fields={"source": "test"},
    )


@pytest.mark.asyncio
async def test_sliding_window_should_compress_ignores_already_compressed_observations():
    tokenizer = _FakeTokenizer()
    manager = SlidingWindowContextManager(
        compress_when_m_observations=2,
        keep_last_n_observations=1,
        tokenizer=tokenizer,
    )
    state = _build_state(
        prompt_text="PROMPT",
        response_text=("<tool_response>[Compressed]</tool_response><tool_response>obs2</tool_response>"),
        messages=[
            {"role": "user", "content": "prompt"},
            {"role": "tool", "content": "[Compressed]"},
            {"role": "tool", "content": "obs2"},
        ],
    )

    assert not await manager.should_compress(state)


@pytest.mark.asyncio
async def test_sliding_window_compress_rewrites_messages_and_response_segment():
    tokenizer = _FakeTokenizer()
    manager = SlidingWindowContextManager(
        compress_when_m_observations=3,
        keep_last_n_observations=1,
        tokenizer=tokenizer,
    )
    response_text = (
        "<tool_response>obs1</tool_response><tool_response>obs2</tool_response><tool_response>obs3</tool_response>"
    )
    state = _build_state(
        prompt_text="PROMPT",
        response_text=response_text,
        messages=[
            {"role": "user", "content": "prompt"},
            {"role": "tool", "content": "obs1"},
            {"role": "tool", "content": "obs2"},
            {"role": "tool", "content": "obs3"},
        ],
        response_logprobs=[0.1] * len(response_text),
        routed_experts="stale-routes",
    )

    compressed_state = await manager.compress(state)
    compressed_response_ids = compressed_state.trajectory_ids[-len(compressed_state.response_mask) :]
    compressed_response_text = tokenizer.decode(compressed_response_ids)

    assert compressed_response_text == (
        "<tool_response>[Compressed]</tool_response>"
        "<tool_response>[Compressed]</tool_response>"
        "<tool_response>obs3</tool_response>"
    )
    assert compressed_state.messages[1]["content"] == "[Compressed]"
    assert compressed_state.messages[2]["content"] == "[Compressed]"
    assert compressed_state.messages[3]["content"] == "obs3"
    assert compressed_state.response_mask == [0] * len(compressed_response_ids)
    assert compressed_state.response_logprobs == [0.0] * len(compressed_response_ids)
    assert compressed_state.routed_experts is None


@pytest.mark.asyncio
async def test_sliding_window_check_and_compress_returns_false_below_threshold():
    manager = SlidingWindowContextManager(
        compress_when_m_observations=2,
        keep_last_n_observations=1,
        tokenizer=_FakeTokenizer(),
    )
    state = _build_state(
        prompt_text="PROMPT",
        response_text="<tool_response>obs1</tool_response>",
        messages=[
            {"role": "user", "content": "prompt"},
            {"role": "tool", "content": "obs1"},
        ],
    )

    next_state, compressed = await manager.check_and_compress(state)

    assert next_state == state
    assert not compressed


@pytest.mark.asyncio
async def test_sliding_window_compress_raises_when_no_new_observation_is_removed():
    tokenizer = _FakeTokenizer()
    manager = SlidingWindowContextManager(
        compress_when_m_observations=2,
        keep_last_n_observations=1,
        tokenizer=tokenizer,
    )
    state = _build_state(
        prompt_text="PROMPT",
        response_text=("<tool_response>[Compressed]</tool_response><tool_response>obs2</tool_response>"),
        messages=[
            {"role": "user", "content": "prompt"},
            {"role": "tool", "content": "[Compressed]"},
            {"role": "tool", "content": "obs2"},
        ],
    )

    with pytest.raises(ValueError, match="removed zero observations unexpectedly"):
        await manager.compress(state)


@pytest.mark.asyncio
async def test_summarizer_should_compress_only_checks_current_generated_tokens():
    tokenizer = _FakeTokenizer()
    old_summary = "<summary>previous summary</summary>"
    current_generation = "new response without summary"
    manager = SummarizerContextManager(tokenizer=tokenizer)
    state = _build_state(
        prompt_text="PROMPT",
        response_text=old_summary + current_generation,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": old_summary},
        ],
        response_mask=[0] * len(old_summary) + [1] * len(current_generation),
    )

    assert not await manager.should_compress(state)


@pytest.mark.asyncio
async def test_summarizer_compress_keeps_last_summary_when_multiple_exist():
    tokenizer = _FakeTokenizer()
    response_text = "thinking...<summary>old summary</summary>more thinking...<summary>new summary</summary>"
    manager = SummarizerContextManager(tokenizer=tokenizer)
    state = _build_state(
        prompt_text="PROMPT",
        response_text=response_text,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ],
    )

    compressed_state = await manager.compress(state)
    assert compressed_state.messages[-1]["content"] == "<summary>new summary</summary>"


@pytest.mark.asyncio
async def test_summarizer_compress_keeps_original_prompt_and_last_summary():
    tokenizer = _FakeTokenizer()
    response_text = "thinking...<summary>new summary</summary>"
    manager = SummarizerContextManager(tokenizer=tokenizer)
    state = _build_state(
        prompt_text="PROMPT",
        response_text=response_text,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "intermediate reasoning"},
            {"role": "tool", "content": "tool observation"},
        ],
        response_logprobs=[0.1] * len(response_text),
        routed_experts="stale-routes",
    )

    compressed_state = await manager.compress(state)
    summary_text = "<summary>new summary</summary>"
    summary_ids = tokenizer.encode(summary_text)

    assert compressed_state.messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": summary_text},
    ]
    assert compressed_state.trajectory_ids[-len(summary_ids) :] == summary_ids
    assert compressed_state.response_mask == [0] * len(summary_ids)
    assert compressed_state.response_logprobs == [0.0] * len(summary_ids)
    assert compressed_state.routed_experts is None


@pytest.mark.asyncio
async def test_summarizer_compress_raises_when_summary_is_missing():
    manager = SummarizerContextManager(tokenizer=_FakeTokenizer())
    state = _build_state(
        prompt_text="PROMPT",
        response_text="plain response without summary",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ],
    )

    with pytest.raises(ValueError, match="expected a <summary> block"):
        await manager.compress(state)
