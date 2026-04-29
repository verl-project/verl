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

from __future__ import annotations

import json
from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, DictConfigWrap
from verl.experimental.agent_loop.agent_loop_with_context_management import (
    SummarizerAgentLoop,
    ToolSlidingWindowAgentLoop,
)
from verl.experimental.agent_loop.context_manager import ContextState
from verl.tools.schemas import ToolResponse
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeTokenizer:
    """Char-level tokenizer mock for deterministic unit tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int] | str:
        del tools, kwargs
        parts = []
        for message in messages:
            if message["role"] == "tool":
                parts.append(f"<tool_response>{message['content']}</tool_response>")
            else:
                parts.append(f"<{message['role']}>{message['content']}")
        text = "".join(parts)
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        return self.encode(text)


class _QueuedServerManager:
    """Minimal fake server manager that returns pre-seeded responses in order.

    Pops one response string per generate() call and records each call in self.calls
    so tests can inspect the prompt_ids passed to the model.
    """

    def __init__(self, tokenizer: _FakeTokenizer, responses: list[str]):
        self._tokenizer = tokenizer
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del sampling_params, image_data, video_data
        if not self._responses:
            raise AssertionError("No fake response left for _QueuedServerManager.generate().")

        response_text = self._responses.pop(0)
        response_ids = self._tokenizer.encode(response_text)
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids), "response_text": response_text})
        return TokenOutput(
            token_ids=response_ids,
            log_probs=[0.0] * len(response_ids),
            num_preempted=0,
        )


def _build_summarizer_loop(
    *, responses: list[str], max_context_compressions: int = 4
) -> tuple[SummarizerAgentLoop, _FakeTokenizer]:
    """Build a summarizer agent loop with deterministic fake dependencies for unit tests."""

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 128, "response_length": 256},
                "model": {},
            },
            "data": {"apply_chat_template_kwargs": {}},
        }
    )
    tokenizer = _FakeTokenizer()
    loop = SummarizerAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=_QueuedServerManager(tokenizer, responses),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        max_context_compressions=max_context_compressions,
    )
    return loop, tokenizer


class _FakeTextTool:
    """Small async text tool used by ToolSlidingWindowAgentLoop tests."""

    def __init__(
        self,
        *,
        responses: Optional[dict[str, ToolResponse]] = None,
        rewards: Optional[dict[str, float]] = None,
    ):
        self.responses = responses or {}
        self.rewards = rewards or {}
        self.created_with: list[dict[str, Any]] = []
        self.executed_with: list[dict[str, Any]] = []
        self.released: list[str] = []

    async def create(self, create_kwargs: dict[str, Any]):
        self.created_with.append(dict(create_kwargs))
        return f"instance-{len(self.created_with)}", {}

    async def execute(self, instance_id: str, parameters: dict[str, Any]):
        del instance_id
        self.executed_with.append(dict(parameters))
        value = str(parameters.get("value", ""))
        response = self.responses.get(value, ToolResponse(text=f"obs:{value}"))
        reward = self.rewards.get(value, 0.0)
        return response, reward, {}

    async def release(self, instance_id: str):
        self.released.append(instance_id)


def _build_tool_loop(
    *,
    responses: list[str],
    tool: Optional[_FakeTextTool] = None,
    response_length: int = 4096,
    max_context_compressions: int = 4,
    compress_when_m_observations: int = 16,
    keep_last_n_observations: int = 2,
    max_assistant_turns: Optional[int] = None,
    max_user_turns: Optional[int] = None,
    interaction_config_path: Optional[str] = None,
) -> tuple[ToolSlidingWindowAgentLoop, _FakeTokenizer, _FakeTextTool]:
    """Build a ToolSlidingWindowAgentLoop with fake server/parser-compatible tool calls."""

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 512,
                    "response_length": response_length,
                    "multi_turn": {
                        "format": "hermes",
                        "tool_config_path": None,
                        "interaction_config_path": interaction_config_path,
                        "max_parallel_calls": 1,
                        "max_assistant_turns": max_assistant_turns,
                        "max_user_turns": max_user_turns,
                        "max_tool_response_length": 256,
                        "tool_response_truncate_side": "middle",
                    },
                },
                "model": {},
            },
            "data": {"apply_chat_template_kwargs": {}},
        }
    )
    tokenizer = _FakeTokenizer()
    loop = ToolSlidingWindowAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=_QueuedServerManager(tokenizer, responses),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        max_context_compressions=max_context_compressions,
        compress_when_m_observations=compress_when_m_observations,
        keep_last_n_observations=keep_last_n_observations,
    )
    tool = tool or _FakeTextTool()
    loop.tools = {"lookup": tool}
    return loop, tokenizer, tool


def _tool_call(value: str, *, name: str = "lookup") -> str:
    payload = {"name": name, "arguments": {"value": value}}
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def _tool_response_ids(tokenizer: _FakeTokenizer, text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "tool", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
    )


def _build_expected_summary_ids(tokenizer: _FakeTokenizer, summary_text: str) -> list[int]:
    """Return the token ids that the loop prepends after summarization compression.

    Mirrors SummarizerContextManager: apply_chat_template on the summary assistant
    message with additional generation tokens, then strip the system-prompt prefix.
    """
    system_prompt_ids = initialize_system_prompt(tokenizer)
    summary_ids = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": summary_text}],
        add_generation_prompt=True,
        tokenize=True,
    )
    return summary_ids[len(system_prompt_ids) :]


def test_summarizer_agent_loop_rejects_negative_max_context_compressions():
    # Passing a negative compression cap should raise ValueError at construction time.
    with pytest.raises(ValueError, match="max_context_compressions must be non-negative"):
        _build_summarizer_loop(responses=["hello"], max_context_compressions=-1)


@pytest.mark.asyncio
async def test_build_output_from_state_handles_empty_response():
    # When response_mask is empty, the entire trajectory should become prompt_ids
    # and response_ids / response_mask should both be empty lists.
    loop, _ = _build_summarizer_loop(responses=[])
    state = ContextState(
        messages=[{"role": "user", "content": "hi"}],
        trajectory_ids=[1, 2, 3],
        response_mask=[],
        response_logprobs=[],
        metrics=AgentLoopMetrics(),
        extra_fields={"source": "test"},
    )

    output = loop._build_output_from_state(state)

    assert output.prompt_ids == [1, 2, 3]
    assert output.response_ids == []
    assert output.response_mask == []
    assert output.extra_fields["source"] == "test"
    assert output.extra_fields["turn_scores"] == []
    assert output.extra_fields["tool_rewards"] == []


@pytest.mark.asyncio
async def test_summarizer_agent_loop_run_returns_multiple_outputs_after_summary_compression():
    # First generation contains a <summary>; the loop compresses and generates again.
    # Verifies: two outputs are returned, both calls share the same request_id,
    # the second output starts with the summary token ids (mask=0) followed by the
    # new generation (mask=1).
    summary_text = "<summary>compressed summary</summary>"
    first_response = f"thinking...{summary_text}"
    second_response = "final answer"
    raw_prompt = [{"role": "user", "content": "hello"}]
    loop, tokenizer = _build_summarizer_loop(
        responses=[first_response, second_response],
        max_context_compressions=1,
    )

    outputs = await loop.run(sampling_params={}, raw_prompt=raw_prompt)

    assert len(outputs) == 2
    assert len(loop.server_manager.calls) == 2
    assert loop.server_manager.calls[0]["request_id"] == loop.server_manager.calls[1]["request_id"]

    first_output_text = tokenizer.decode(outputs[0].response_ids)
    second_output_text = tokenizer.decode(outputs[1].response_ids)
    summary_ids = _build_expected_summary_ids(tokenizer, summary_text)

    assert first_output_text == first_response
    assert second_output_text == tokenizer.decode(summary_ids) + second_response
    assert outputs[0].response_mask == [1] * len(outputs[0].response_ids)
    assert outputs[1].response_mask[: len(summary_ids)] == [0] * len(summary_ids)
    assert outputs[1].response_mask[len(summary_ids) :] == [1] * len(tokenizer.encode(second_response))


@pytest.mark.asyncio
async def test_summarizer_agent_loop_run_returns_single_output_without_summary():
    # No <summary> in the response means no compression; run() returns exactly one output
    # with all tokens marked as generated (response_mask all-ones).
    loop, tokenizer = _build_summarizer_loop(responses=["plain final answer"], max_context_compressions=4)

    outputs = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "hello"}])

    assert len(outputs) == 1
    assert tokenizer.decode(outputs[0].response_ids) == "plain final answer"
    assert outputs[0].response_mask == [1] * len(outputs[0].response_ids)


@pytest.mark.asyncio
async def test_summarizer_agent_loop_run_respects_zero_max_context_compressions():
    # max_context_compressions=0 means compression is never applied even if a <summary>
    # is present; run() stops after the first generation with a single output.
    summary_text = "<summary>compressed summary</summary>"
    first_response = f"thinking...{summary_text}"
    loop, tokenizer = _build_summarizer_loop(responses=[first_response], max_context_compressions=0)

    outputs = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "hello"}])

    assert len(outputs) == 1
    assert len(loop.server_manager.calls) == 1
    assert tokenizer.decode(outputs[0].response_ids) == first_response


@pytest.mark.asyncio
async def test_summarizer_agent_loop_run_supports_multiple_compressions_until_cap():
    # Two consecutive compressions (cap=2): each compressed output starts with the
    # corresponding summary ids (mask=0) followed by the new generation (mask=1).
    # Verifies that the loop chains compressions correctly up to the cap.
    summary1 = "<summary>summary 1</summary>"
    summary2 = "<summary>summary 2</summary>"
    responses = [
        f"step1...{summary1}",
        f"step2...{summary2}",
        "final answer",
    ]
    raw_prompt = [{"role": "user", "content": "hello"}]
    loop, tokenizer = _build_summarizer_loop(responses=responses, max_context_compressions=2)

    outputs = await loop.run(sampling_params={}, raw_prompt=raw_prompt)

    assert len(outputs) == 3
    assert len(loop.server_manager.calls) == 3
    summary1_ids = _build_expected_summary_ids(tokenizer, summary1)
    summary2_ids = _build_expected_summary_ids(tokenizer, summary2)
    assert tokenizer.decode(outputs[0].response_ids) == responses[0]
    assert tokenizer.decode(outputs[1].response_ids) == tokenizer.decode(summary1_ids) + responses[1]
    assert tokenizer.decode(outputs[2].response_ids) == tokenizer.decode(summary2_ids) + responses[2]


def test_tool_sliding_window_agent_loop_rejects_invalid_constructor_args():
    # The loop is intentionally text-only and does not support interaction callbacks.
    with pytest.raises(ValueError, match="max_context_compressions must be non-negative"):
        _build_tool_loop(responses=["unused"], max_context_compressions=-1)

    with pytest.raises(ValueError, match="does not support interaction_config_path"):
        _build_tool_loop(responses=["unused"], interaction_config_path="/tmp/interaction.json")


@pytest.mark.asyncio
async def test_tool_sliding_window_agent_loop_rejects_non_text_prompt_messages():
    loop, _, _ = _build_tool_loop(responses=["unused"])

    with pytest.raises(ValueError, match="only supports string message content"):
        await loop.run(
            sampling_params={},
            raw_prompt=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )

    assert loop.server_manager.calls == []


@pytest.mark.asyncio
async def test_tool_sliding_window_agent_loop_runs_tool_round_without_reparsing_history():
    # The first assistant response contains a tool call; the second one is a final
    # answer. If the parser accidentally scans the whole history, it would see the
    # old tool call again and try to execute the tool a second time.
    tool_call = _tool_call("first")
    final_response = "final answer"
    tool = _FakeTextTool(rewards={"first": 0.5})
    loop, tokenizer, tool = _build_tool_loop(
        responses=[tool_call, final_response],
        tool=tool,
        compress_when_m_observations=2,
        keep_last_n_observations=1,
    )

    outputs = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "lookup first"}])

    assert len(outputs) == 1
    assert len(loop.server_manager.calls) == 2
    assert loop.server_manager.calls[0]["request_id"] == loop.server_manager.calls[1]["request_id"]
    assert tool.executed_with == [{"value": "first"}]
    assert tool.released == ["instance-1"]

    tool_response_ids = _tool_response_ids(tokenizer, "obs:first")
    expected_response_text = tool_call + tokenizer.decode(tool_response_ids) + final_response
    expected_response_mask = (
        [1] * len(tokenizer.encode(tool_call))
        + [0] * len(tool_response_ids)
        + [1] * len(tokenizer.encode(final_response))
    )

    output = outputs[0]
    assert tokenizer.decode(output.response_ids) == expected_response_text
    assert output.response_mask == expected_response_mask
    assert output.extra_fields["tool_rewards"] == [0.5]
    assert output.extra_fields["session_tool_rewards"] == [0.5]
    assert output.extra_fields["context_compression_count"] == 0
    assert output.extra_fields["agent_loop_impl"] == "ToolSlidingWindowAgentLoop"


@pytest.mark.asyncio
async def test_tool_sliding_window_agent_loop_recompresses_without_recounting_replaced_tool_responses():
    first_tool_call = _tool_call("first")
    second_tool_call = _tool_call("second")
    third_tool_call = _tool_call("third")
    first_assistant = f"reasoning1: inspect first. {first_tool_call}"
    second_assistant = f"reasoning2: inspect second. {second_tool_call}"
    third_assistant = f"reasoning3: inspect third. {third_tool_call}"
    final_response = "final output"
    tool = _FakeTextTool(rewards={"first": 0.25, "second": 0.75, "third": 1.25})
    loop, tokenizer, tool = _build_tool_loop(
        responses=[first_assistant, second_assistant, third_assistant, final_response],
        tool=tool,
        max_context_compressions=2,
        compress_when_m_observations=2,
        keep_last_n_observations=1,
    )

    outputs = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "lookup three times"}])

    assert len(outputs) == 3
    assert len(loop.server_manager.calls) == 4
    assert tool.executed_with == [{"value": "first"}, {"value": "second"}, {"value": "third"}]

    prompt_text = tokenizer.decode(outputs[0].prompt_ids)
    first_tool_response_ids = _tool_response_ids(tokenizer, "obs:first")
    second_tool_response_ids = _tool_response_ids(tokenizer, "obs:second")
    third_tool_response_ids = _tool_response_ids(tokenizer, "obs:third")
    compressed_tool_response_text = tokenizer.decode(_tool_response_ids(tokenizer, "[Compressed]"))
    first_tool_response_text = tokenizer.decode(first_tool_response_ids)
    second_tool_response_text = tokenizer.decode(second_tool_response_ids)
    third_tool_response_text = tokenizer.decode(third_tool_response_ids)

    first_precompression_text = (
        first_assistant + first_tool_response_text + second_assistant + second_tool_response_text
    )
    first_precompression_mask = (
        [1] * len(tokenizer.encode(first_assistant))
        + [0] * len(first_tool_response_ids)
        + [1] * len(tokenizer.encode(second_assistant))
        + [0] * len(second_tool_response_ids)
    )
    first_output = outputs[0]
    first_full_text = tokenizer.decode(first_output.prompt_ids + first_output.response_ids)
    assert first_full_text == prompt_text + first_precompression_text
    assert first_output.response_mask == first_precompression_mask
    assert first_output.extra_fields["tool_rewards"] == [0.25, 0.75]
    assert first_output.extra_fields["session_tool_rewards"] == [0.25, 0.75]
    assert first_output.extra_fields["context_compression_count"] == 0

    once_compressed_prefix_text = (
        first_assistant + compressed_tool_response_text + second_assistant + second_tool_response_text
    )
    second_precompression_text = once_compressed_prefix_text + third_assistant + third_tool_response_text
    second_precompression_mask = (
        [0] * len(tokenizer.encode(once_compressed_prefix_text))
        + [1] * len(tokenizer.encode(third_assistant))
        + [0] * len(third_tool_response_ids)
    )
    second_output = outputs[1]
    assert tokenizer.decode(second_output.response_ids) == second_precompression_text
    assert second_output.response_mask == second_precompression_mask
    assert "obs:first" not in second_precompression_text
    assert "obs:second" in second_precompression_text
    assert "obs:third" in second_precompression_text
    assert second_output.extra_fields["tool_rewards"] == [1.25]
    assert second_output.extra_fields["session_tool_rewards"] == [0.25, 0.75, 1.25]
    assert second_output.extra_fields["context_compression_count"] == 1

    twice_compressed_prefix_text = (
        first_assistant
        + compressed_tool_response_text
        + second_assistant
        + compressed_tool_response_text
        + third_assistant
        + third_tool_response_text
    )
    final_output = outputs[2]
    assert tokenizer.decode(final_output.response_ids) == twice_compressed_prefix_text + final_response
    assert final_output.response_mask == [0] * len(tokenizer.encode(twice_compressed_prefix_text)) + [1] * len(
        tokenizer.encode(final_response)
    )
    assert final_output.extra_fields["tool_rewards"] == []
    assert final_output.extra_fields["session_tool_rewards"] == [0.25, 0.75, 1.25]
    assert final_output.extra_fields["context_compression_count"] == 2

    first_compressed_prompt = tokenizer.decode(loop.server_manager.calls[2]["prompt_ids"], skip_special_tokens=False)
    second_compressed_prompt = tokenizer.decode(loop.server_manager.calls[3]["prompt_ids"], skip_special_tokens=False)
    assert first_compressed_prompt == prompt_text + once_compressed_prefix_text
    assert second_compressed_prompt == prompt_text + twice_compressed_prefix_text
    assert first_compressed_prompt.count("[Compressed]") == 1
    assert second_compressed_prompt.count("[Compressed]") == 2
    assert "obs:first" not in second_compressed_prompt
    assert "obs:second" not in second_compressed_prompt
    assert "obs:third" in second_compressed_prompt


@pytest.mark.asyncio
async def test_tool_sliding_window_agent_loop_rejects_multimodal_tool_response():
    tool = _FakeTextTool(responses={"image": ToolResponse(text="obs:image", image=["fake-image"])})
    loop, _, tool = _build_tool_loop(responses=[_tool_call("image")], tool=tool)

    with pytest.raises(ValueError, match="only supports text tool responses"):
        await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "call image tool"}])

    assert tool.executed_with == [{"value": "image"}]
    assert tool.released == ["instance-1"]
