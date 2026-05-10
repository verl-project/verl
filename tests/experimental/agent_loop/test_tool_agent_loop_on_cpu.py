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

"""CPU unit tests for ToolAgentLoop state transitions."""

import unittest
from types import SimpleNamespace

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.tools.schemas import ToolResponse


def _make_agent_data() -> AgentData:
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="test-request",
        tools_kwargs={},
    )
    agent_data.prompt_ids = [11, 12, 101, 102]
    agent_data.response_mask = [1, 1]
    agent_data.response_logprobs = [0.1, 0.2]
    agent_data.tool_calls = [SimpleNamespace(name="lookup", arguments="{}")]
    return agent_data


def _make_tool_agent_loop(tool_response_ids: list[int], response_length: int) -> ToolAgentLoop:
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.max_parallel_calls = 1
    loop.response_length = response_length
    loop.tool_parser_name = "hermes"
    loop.processor = None

    async def _call_tool(tool_call, tools_kwargs, agent_data):
        return ToolResponse(text="tool result"), None, {}

    async def apply_chat_template(*args, **kwargs):
        return tool_response_ids

    loop._call_tool = _call_tool
    loop.apply_chat_template = apply_chat_template
    return loop


class TestToolAgentLoopStateTransitions(unittest.IsolatedAsyncioTestCase):
    async def test_tool_response_exact_fit_is_appended_before_termination(self):
        loop = _make_tool_agent_loop(tool_response_ids=[201, 202], response_length=4)
        agent_data = _make_agent_data()

        state = await loop._handle_processing_tools_state(agent_data)

        assert state is AgentState.TERMINATED
        assert agent_data.prompt_ids == [11, 12, 101, 102, 201, 202]
        assert agent_data.response_mask == [1, 1, 0, 0]
        assert agent_data.response_logprobs == [0.1, 0.2, 0.0, 0.0]
        assert agent_data.user_turns == 1

    async def test_tool_response_overflow_still_terminates_without_appending(self):
        loop = _make_tool_agent_loop(tool_response_ids=[201, 202, 203], response_length=4)
        agent_data = _make_agent_data()

        state = await loop._handle_processing_tools_state(agent_data)

        assert state is AgentState.TERMINATED
        assert agent_data.prompt_ids == [11, 12, 101, 102]
        assert agent_data.response_mask == [1, 1]
        assert agent_data.response_logprobs == [0.1, 0.2]
        assert agent_data.user_turns == 0


if __name__ == "__main__":
    unittest.main()
