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
"""Truncation accumulates across tool turns and includes the agent's response budget."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.workers.rollout.replica import TokenOutput


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flags,cap,expected",
    [
        ([False, False], 8, False),
        ([False, True], 8, True),
        ([True, False], 8, True),
        ([False, None], 8, None),
        ([None, True], 8, True),
        ([None, False], 8, None),
        ([False, False], 2, True),
    ],
)
async def test_tool_turn_truncation_is_preserved(flags, cap, expected):
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.response_length = cap
    loop.max_assistant_turns = loop.max_user_turns = None
    loop.tools = {}
    loop.tool_schemas = []
    loop.process_multi_modal_info = AsyncMock(return_value={})
    loop._get_mm_processor_kwargs = lambda _: {}
    loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            side_effect=[
                TokenOutput(token_ids=[10], is_truncated=flags[0]),
                TokenOutput(token_ids=[11], is_truncated=flags[1]),
            ]
        )
    )
    loop.tool_parser = SimpleNamespace(
        stop_token_ids=[],
        extract_tool_calls=AsyncMock(
            side_effect=[
                ("tool call", [object()]),
                ("done", []),
            ]
        ),
    )
    loop._build_assistant_message = lambda content, _: {"role": "assistant", "content": content}

    async def pending(data, sampling_params):
        data.prompt_ids = [1, 2]
        return AgentState.GENERATING

    async def tools(data):
        return AgentState.GENERATING

    async def merge(prompt_ids, token_ids, response_mask, response_logprobs, **kwargs):
        return SimpleNamespace(token_ids=prompt_ids + token_ids), response_mask + [1] * len(token_ids), None

    loop._handle_pending_state = pending
    loop._handle_processing_tools_state = tools
    loop.ct_merge_assistant_token = merge
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}])
    assert loop.server_manager.generate.await_count == 2
    assert output.extra_fields["response_truncated"] is expected


@pytest.mark.asyncio
async def test_tool_observation_that_exhausts_budget_marks_trajectory_truncated():
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.response_length = 4
    loop.max_parallel_calls = 1
    loop.tool_schemas = []
    loop._assert_mm_supported = lambda _: None
    loop.ct_merge_context_msg = AsyncMock(
        return_value=(SimpleNamespace(token_ids=[1, 2, 10, 20, 21, 22]), [1, 0, 0, 0], None)
    )
    data = AgentData(
        messages=[],
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        metrics={},
        request_id="req",
        tools_kwargs={},
    )
    data.prompt_ids = [1, 2, 10]
    data.response_mask = [1]
    # No external tool invocation is needed: the tokenizer result alone exhausts the budget.
    state = await loop._handle_processing_tools_state(data)
    assert state == AgentState.TERMINATED
    assert data.is_truncated is True
    assert data.response_mask == [1]
