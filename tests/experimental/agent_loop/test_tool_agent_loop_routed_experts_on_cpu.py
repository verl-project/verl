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
"""CPU regression tests for routed-expert alignment across tool-agent turns."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.tool_agent_loop import (
    AgentState,
    ToolAgentLoop,
    _merge_routed_experts,
    _trim_routed_experts_to_prefix,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse
from verl.utils.tokenizer.continuous_token import MergeResult
from verl.workers.rollout.replica import TokenOutput

LAYERS, TOPK = 2, 2


def _routing(markers: list[int], *, read_only: bool = False) -> np.ndarray:
    array = np.repeat(np.asarray(markers, dtype=np.int32), LAYERS * TOPK).reshape(-1, LAYERS, TOPK)
    if read_only:
        return np.frombuffer(array.tobytes(), dtype=np.int32).reshape(-1, LAYERS, TOPK)
    return array


def _markers(routed_experts: Any) -> list[int]:
    return [int(value) for value in routed_experts[:, 0, 0]]


def test_merge_preserves_previous_turn_and_appends_newly_covered_suffix() -> None:
    previous = _routing([10, 11, 12, 13], read_only=True)
    current = _routing([20, 21, 22, 23, 24, 25, 26, 27], read_only=True)

    merged = _merge_routed_experts(previous, current, prompt_length=7)

    assert isinstance(merged, np.ndarray)
    assert _markers(merged) == [10, 11, 12, 13, 24, 25, 26, 27]
    assert merged.flags.writeable


def test_merge_supports_torch_routing_without_changing_the_output_type() -> None:
    previous = torch.from_numpy(_routing([10, 11]))
    current = torch.from_numpy(_routing([20, 21, 22, 23]))

    merged = _merge_routed_experts(previous, current, prompt_length=3)

    assert isinstance(merged, torch.Tensor)
    assert _markers(merged) == [10, 11, 22, 23]


def test_merge_keeps_existing_routes_when_current_turn_has_none() -> None:
    previous = _routing([10, 11])

    assert _merge_routed_experts(previous, None, prompt_length=3) is previous
    assert _merge_routed_experts(None, None, prompt_length=3) is None


def test_merge_rejects_misaligned_snapshots() -> None:
    previous = _routing([10, 11, 12, 13])

    with pytest.raises(ValueError, match="exceed the current generation prompt"):
        _merge_routed_experts(previous, _routing([20, 21, 22, 23]), prompt_length=3)

    with pytest.raises(ValueError, match="shorter than the already captured prefix"):
        _merge_routed_experts(previous, _routing([20, 21, 22]), prompt_length=4)

    with pytest.raises(ValueError, match="layer/top-k dimensions changed"):
        _merge_routed_experts(previous, np.zeros((5, LAYERS + 1, TOPK), dtype=np.int32), prompt_length=4)


def test_trim_only_drops_routing_rows_that_were_already_covered() -> None:
    fully_covered = _routing([10, 11, 12, 13, 14])
    trimmed = _trim_routed_experts_to_prefix(fully_covered, prefix_length=4)
    assert _markers(trimmed) == [10, 11, 12, 13]

    final_token_not_covered = _routing([10, 11, 12, 13])
    unchanged = _trim_routed_experts_to_prefix(final_token_not_covered, prefix_length=4)
    assert unchanged is final_token_not_covered


class _SequenceServer:
    def __init__(self, outputs: list[TokenOutput]):
        self._outputs = iter(outputs)
        self.prompts_seen: list[list[int]] = []

    async def generate(self, request_id: str, *, prompt_ids: list[int], **kwargs: Any) -> TokenOutput:
        del request_id, kwargs
        self.prompts_seen.append(list(prompt_ids))
        return next(self._outputs)


class _NoToolCallsParser:
    stop_token_ids: list[int] = []

    async def extract_tool_calls(self, response_ids: list[int], tools: list[Any]) -> tuple[str, list[Any]]:
        del response_ids, tools
        return "", []


@pytest.mark.asyncio
async def test_generating_state_preserves_routes_across_tool_turns() -> None:
    server = _SequenceServer(
        [
            TokenOutput(
                token_ids=[101, 102],
                routed_experts=_routing([10, 11, 12, 13]),
                extra_fields={"max_global_steps": 1},
            ),
            TokenOutput(
                token_ids=[103, 104],
                routed_experts=_routing([20, 21, 22, 23, 24, 25, 26, 27]),
                extra_fields={"max_global_steps": 2},
            ),
        ]
    )
    loop = SimpleNamespace(
        server_manager=server,
        tool_parser=_NoToolCallsParser(),
        enable_continuous_token=False,
        response_length=64,
        max_assistant_turns=0,
        max_user_turns=0,
        tools={},
    )
    agent_data = SimpleNamespace(
        request_id="request-0",
        prompt_ids=[1, 2, 3],
        metrics={},
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs={},
        extra_fields={},
        assistant_turns=0,
        response_ids=[],
        response_mask=[],
        response_logprobs=[],
        routed_experts=None,
        user_turns=0,
        tool_calls=[],
        messages=[],
    )

    await ToolAgentLoop._handle_generating_state(loop, agent_data, {}, ignore_termination=True)
    agent_data.prompt_ids.extend([201, 202])
    agent_data.response_mask.extend([0, 0])
    await ToolAgentLoop._handle_generating_state(loop, agent_data, {}, ignore_termination=True)

    assert server.prompts_seen == [[1, 2, 3], [1, 2, 3, 101, 102, 201, 202]]
    assert _markers(agent_data.routed_experts) == [10, 11, 12, 13, 24, 25, 26, 27]


@pytest.mark.asyncio
async def test_continuous_token_prefix_removal_trims_captured_routes() -> None:
    async def call_tool(
        tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: SimpleNamespace
    ) -> tuple[ToolResponse, None, dict[str, Any]]:
        del tool_call, tools_kwargs, agent_data
        return ToolResponse(text="result"), None, {}

    async def merge_non_assistant(
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        response_mask: list[int],
        response_logprobs: None,
        *,
        tools: list[Any],
    ) -> tuple[MergeResult, list[int], None]:
        del previous_messages, updated_messages, runtime_token_ids, response_mask, response_logprobs, tools
        return (
            MergeResult(
                token_ids=[1, 2, 3, 4, 90, 91],
                appended_token_count=1,
                kind="non_assistant",
                inserted_token_ids=[90],
                removed_prefix_token_count=1,
            ),
            [1, 0, 0],
            None,
        )

    loop = SimpleNamespace(
        max_parallel_calls=1,
        processor=None,
        enable_continuous_token=True,
        response_length=64,
        tool_parser_name="qwen3_coder",
        tool_schemas=[],
        _call_tool=call_tool,
        ct_merge_non_assistant_msg=merge_non_assistant,
    )
    agent_data = SimpleNamespace(
        messages=[{"role": "user", "content": "question"}, {"role": "assistant", "content": "call"}],
        tool_calls=[FunctionCall(name="lookup", arguments="{}")],
        tools_kwargs={},
        metrics={},
        tool_rewards=[],
        prompt_ids=[1, 2, 3, 4, 5],
        response_mask=[1, 1],
        response_logprobs=[],
        image_data=None,
        user_turns=0,
        routed_experts=_routing([10, 11, 12, 13, 14]),
    )

    state = await ToolAgentLoop._handle_processing_tools_state(loop, agent_data)

    assert state == AgentState.GENERATING
    assert agent_data.prompt_ids == [1, 2, 3, 4, 90, 91]
    assert _markers(agent_data.routed_experts) == [10, 11, 12, 13]
