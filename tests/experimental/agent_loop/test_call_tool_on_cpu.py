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

"""Unit tests for ToolAgentLoop._call_tool error handling (no GPU required).

Tests that malformed tool calls return specific, actionable error messages
instead of generic exception strings.
"""

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from verl.tools.schemas import ToolResponse
from verl.workers.rollout.request_scheduling import (
    RolloutRequestContext,
    RolloutRequestKind,
)


@dataclass
class FakeFunctionCall:
    """Minimal FunctionCall for testing."""

    name: str
    arguments: str


@dataclass
class FakeAgentData:
    """Minimal AgentData for testing."""

    tools_kwargs: dict = field(default_factory=dict)


class FakeTool:
    """A fake tool that succeeds."""

    def __init__(self, name: str):
        self.name = name

    async def create(self, create_kwargs=None):
        return "instance_1", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        return ToolResponse(text=f"OK: {parameters}"), 1.0, {}

    async def release(self, instance_id):
        pass


class FakeFailingTool(FakeTool):
    """A fake tool that raises during execute."""

    async def execute(self, instance_id, parameters, **kwargs):
        raise RuntimeError("database connection failed")


class FakeLongResponseTool(FakeTool):
    """A fake tool that returns a long response."""

    def __init__(self, name: str, text: str):
        super().__init__(name)
        self.text = text

    async def execute(self, instance_id, parameters, **kwargs):
        return ToolResponse(text=self.text), 1.0, {}


def _make_tool_agent_loop(
    tools: dict[str, Any],
    max_tool_response_length: int = 10000,
    tool_response_truncate_side: str = "left",
):
    """Create a minimal ToolAgentLoop instance with only the fields _call_tool needs."""
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    mock = MagicMock(spec=ToolAgentLoop)
    mock.tools = tools
    mock.max_tool_response_length = max_tool_response_length
    mock.tool_response_truncate_side = tool_response_truncate_side
    # Bind the real _call_tool method to our mock
    mock._call_tool = ToolAgentLoop._call_tool.__get__(mock, ToolAgentLoop)
    return mock


def _make_generation_loop():
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    mock = MagicMock(spec=ToolAgentLoop)
    mock.server_manager = SimpleNamespace(generate=AsyncMock(return_value=object()))
    mock.response_length = 16
    mock._generate = ToolAgentLoop._generate.__get__(mock, ToolAgentLoop)
    return mock


def _make_generation_agent_data(*, assistant_turns: int, prompt_ids: list[int], previous_length: int):
    return SimpleNamespace(
        request_id="trajectory-1",
        assistant_turns=assistant_turns,
        prompt_ids=prompt_ids,
        last_model_sequence_length=previous_length,
        response_mask=[],
        image_data=None,
        video_data=None,
        mm_processor_output=None,
        audio_data=None,
        mm_processor_kwargs={},
    )


class TestCallToolErrorHandling(unittest.IsolatedAsyncioTestCase):
    """Test ToolAgentLoop._call_tool error handling for malformed tool calls."""

    def setUp(self):
        self.tools = {
            "calculator": FakeTool("calculator"),
            "search": FakeTool("search"),
        }
        self.loop = _make_tool_agent_loop(self.tools)
        self.agent_data = FakeAgentData()

    async def test_valid_tool_call(self):
        """Valid tool call should succeed."""
        tool_call = FakeFunctionCall(name="calculator", arguments='{"a": 3, "b": 5}')
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert "OK" in response.text

    async def test_unknown_function_name(self):
        """Unknown function name should list available tools."""
        tool_call = FakeFunctionCall(name="calculater", arguments='{"a": 3}')
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Unknown function" in response.text
        assert "calculater" in response.text
        assert "calculator" in response.text
        assert "search" in response.text

    async def test_invalid_json_arguments(self):
        """Invalid JSON arguments should report parse error."""
        tool_call = FakeFunctionCall(name="calculator", arguments="{a: 3}")
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text
        assert "calculator" in response.text

    async def test_empty_arguments(self):
        """Empty string arguments should report parse error."""
        tool_call = FakeFunctionCall(name="calculator", arguments="")
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text

    async def test_none_arguments(self):
        """None arguments should report error."""
        tool_call = FakeFunctionCall(name="calculator", arguments=None)
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text

    async def test_tool_execution_error(self):
        """Tool execution failure should include tool name in error."""
        tools = {"failing_tool": FakeFailingTool("failing_tool")}
        loop = _make_tool_agent_loop(tools)
        tool_call = FakeFunctionCall(name="failing_tool", arguments='{"query": "test"}')
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "failing_tool" in response.text
        assert "database connection failed" in response.text

    async def test_left_truncation_keeps_response_tail(self):
        """Left truncation should drop the left side and preserve the response tail."""
        tool_response = (
            "Search results for capital of France:\n"
            "1. Lyon is a major city with a long Roman history.\n"
            "2. Marseille is a large port city in southern France.\n"
            "3. The final retrieved snippet says the capital is Paris.\n"
            "Final answer: Paris"
        )
        tools = {"search": FakeLongResponseTool("search", tool_response)}
        loop = _make_tool_agent_loop(tools, max_tool_response_length=19, tool_response_truncate_side="left")
        tool_call = FakeFunctionCall(name="search", arguments="{}")
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert response.text.startswith("(truncated)...")
        assert response.text.endswith("Final answer: Paris")

    async def test_right_truncation_keeps_response_head(self):
        """Right truncation should drop the right side and preserve the response head."""
        tool_response = (
            "Search results for capital of France:\n"
            "1. Lyon is a major city with a long Roman history.\n"
            "2. Marseille is a large port city in southern France.\n"
            "3. The final retrieved snippet says the capital is Paris.\n"
            "Final answer: Paris"
        )
        tools = {"search": FakeLongResponseTool("search", tool_response)}
        loop = _make_tool_agent_loop(tools, max_tool_response_length=19, tool_response_truncate_side="right")
        tool_call = FakeFunctionCall(name="search", arguments="{}")
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert response.text.startswith("Search results")
        assert response.text.endswith("...(truncated)")
        assert "Final answer: Paris" not in response.text


class TestToolAgentRequestScheduling(unittest.IsolatedAsyncioTestCase):
    async def test_generation_sends_admission_metadata_without_backend_priority(self):
        loop = _make_generation_loop()
        agent_data = _make_generation_agent_data(
            assistant_turns=0,
            prompt_ids=[1, 2, 3],
            previous_length=0,
        )

        await loop._generate(agent_data, {"temperature": 0})

        kwargs = loop.server_manager.generate.await_args.kwargs
        assert "priority" not in kwargs
        context = kwargs["request_context"]
        assert context["trajectory_id"] == "trajectory-1"
        assert context["request_kind"] == "fresh"
        assert context["turn_index"] == 0
        assert context["attempt_index"] == 0
        assert context["prompt_tokens"] == 3
        assert context["estimated_uncached_tokens"] == 3
        assert context["expected_output_tokens"] == 16
        assert isinstance(context["enqueued_at"], float)

    async def test_request_context_distinguishes_fresh_and_continuation(self):
        loop = _make_generation_loop()

        fresh = _make_generation_agent_data(
            assistant_turns=0,
            prompt_ids=[1, 2, 3],
            previous_length=0,
        )
        await loop._generate(fresh, {})

        continuation = _make_generation_agent_data(
            assistant_turns=1,
            prompt_ids=[1, 2, 3, 4, 5],
            previous_length=3,
        )
        await loop._generate(continuation, {})

        first_call, second_call = loop.server_manager.generate.await_args_list
        assert "priority" not in first_call.kwargs
        assert first_call.kwargs["request_context"]["request_kind"] == "fresh"
        assert "priority" not in second_call.kwargs
        assert second_call.kwargs["request_context"]["request_kind"] == "continuation"
        assert second_call.kwargs["request_context"]["estimated_uncached_tokens"] == 2

    def test_request_context_serializes_enum_value(self):
        context = RolloutRequestContext(
            trajectory_id="trajectory-1",
            request_kind=RolloutRequestKind.RETRY,
            turn_index=2,
            attempt_index=1,
            prompt_tokens=12,
            estimated_uncached_tokens=4,
            enqueued_at=1.0,
        )

        serialized = context.as_dict()
        assert serialized["request_kind"] == "retry"
        assert RolloutRequestContext.from_dict(serialized) == context


if __name__ == "__main__":
    unittest.main()
