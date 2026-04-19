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
from typing import Any
from unittest.mock import MagicMock

from verl.tools.schemas import ToolResponse


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


def _make_tool_agent_loop(tools: dict[str, Any]):
    """Create a minimal ToolAgentLoop instance with only the fields _call_tool needs."""
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    mock = MagicMock(spec=ToolAgentLoop)
    mock.tools = tools
    mock.max_tool_response_length = 10000
    mock.tool_response_truncate_side = "left"
    # Bind the real _call_tool method to our mock
    mock._call_tool = ToolAgentLoop._call_tool.__get__(mock, ToolAgentLoop)
    return mock


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


if __name__ == "__main__":
    unittest.main()
