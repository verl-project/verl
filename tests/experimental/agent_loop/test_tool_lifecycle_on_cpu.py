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

"""Unit tests for the per-trajectory BaseTool lifecycle in ToolAgentLoop (no GPU required).

A BaseTool instance is created by the first call to that tool in a trajectory, reused by
later calls, and released once when the trajectory ends.
"""

import asyncio
import unittest
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

from verl.tools.schemas import ToolResponse


@dataclass
class FakeFunctionCall:
    name: str
    arguments: str = "{}"


@dataclass
class FakeAgentData:
    tools_kwargs: dict = field(default_factory=dict)
    tool_instances: dict = field(default_factory=dict)
    tool_instance_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CountingTool:
    """Stateful tool: every execute increments a counter kept per instance."""

    def __init__(self, name: str):
        self.name = name
        self.created: list[tuple[str, dict]] = []
        self.released: list[str] = []
        self.counters: dict[str, int] = {}

    async def create(self, create_kwargs=None):
        instance_id = f"{self.name}-{len(self.created)}"
        self.created.append((instance_id, create_kwargs))
        self.counters[instance_id] = 0
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        self.counters[instance_id] += 1
        return ToolResponse(text=str(self.counters[instance_id])), 0.0, {}

    async def release(self, instance_id):
        self.released.append(instance_id)


class FlakyCreateTool(CountingTool):
    """Fails the first create, then behaves normally."""

    async def create(self, create_kwargs=None):
        if not self.created and not getattr(self, "failed_once", False):
            self.failed_once = True
            raise RuntimeError("sandbox unavailable")
        return await super().create(create_kwargs)


class FailingReleaseTool(CountingTool):
    async def release(self, instance_id):
        raise RuntimeError("release failed")


def _make_loop(tools: dict):
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    loop = MagicMock(spec=ToolAgentLoop)
    loop.tools = tools
    loop.tool_schemas = []
    loop.max_tool_response_length = 10000
    loop.tool_response_truncate_side = "left"
    for name in ("_call_tool", "_release_tool_instances", "run"):
        setattr(loop, name, getattr(ToolAgentLoop, name).__get__(loop, ToolAgentLoop))
    return loop


class TestToolLifecyclePerTrajectory(unittest.IsolatedAsyncioTestCase):
    async def test_state_persists_across_calls_in_one_trajectory(self):
        tool = CountingTool("counter")
        loop = _make_loop({"counter": tool})
        agent_data = FakeAgentData()

        first, _, _ = await loop._call_tool(FakeFunctionCall("counter"), {}, agent_data)
        second, _, _ = await loop._call_tool(FakeFunctionCall("counter"), {}, agent_data)

        assert (first.text, second.text) == ("1", "2")
        assert len(tool.created) == 1
        assert tool.released == []

    async def test_parallel_calls_in_one_turn_share_a_single_instance(self):
        tool = CountingTool("counter")
        loop = _make_loop({"counter": tool})
        agent_data = FakeAgentData()

        await asyncio.gather(*[loop._call_tool(FakeFunctionCall("counter"), {}, agent_data) for _ in range(5)])

        assert len(tool.created) == 1
        assert tool.counters["counter-0"] == 5

    async def test_each_trajectory_gets_its_own_instance(self):
        tool = CountingTool("counter")
        loop = _make_loop({"counter": tool})

        await loop._call_tool(FakeFunctionCall("counter"), {}, FakeAgentData())
        await loop._call_tool(FakeFunctionCall("counter"), {}, FakeAgentData())

        assert [instance_id for instance_id, _ in tool.created] == ["counter-0", "counter-1"]

    async def test_create_kwargs_are_passed_once_at_creation(self):
        tool = CountingTool("counter")
        loop = _make_loop({"counter": tool})
        tools_kwargs = {"counter": {"create_kwargs": {"ground_truth": "42"}}}
        agent_data = FakeAgentData(tools_kwargs=tools_kwargs)

        await loop._call_tool(FakeFunctionCall("counter"), tools_kwargs, agent_data)
        await loop._call_tool(FakeFunctionCall("counter"), tools_kwargs, agent_data)

        assert tool.created == [("counter-0", {"ground_truth": "42"})]

    async def test_failed_create_is_retried_by_the_next_call(self):
        tool = FlakyCreateTool("flaky")
        loop = _make_loop({"flaky": tool})
        agent_data = FakeAgentData()

        failed, reward, _ = await loop._call_tool(FakeFunctionCall("flaky"), {}, agent_data)
        recovered, _, _ = await loop._call_tool(FakeFunctionCall("flaky"), {}, agent_data)

        assert reward == 0.0
        assert "sandbox unavailable" in failed.text
        assert recovered.text == "1"
        assert agent_data.tool_instances == {"flaky": "flaky-0"}


class TestReleaseAtTrajectoryEnd(unittest.IsolatedAsyncioTestCase):
    async def test_every_instance_is_released_even_if_one_release_fails(self):
        broken, healthy = FailingReleaseTool("broken"), CountingTool("healthy")
        loop = _make_loop({"broken": broken, "healthy": healthy})
        agent_data = FakeAgentData()
        await loop._call_tool(FakeFunctionCall("broken"), {}, agent_data)
        await loop._call_tool(FakeFunctionCall("healthy"), {}, agent_data)

        await loop._release_tool_instances(agent_data)

        assert healthy.released == ["healthy-0"]

    async def test_instances_are_released_when_the_trajectory_raises(self):
        tool = CountingTool("counter")
        loop = _make_loop({"counter": tool})
        loop.process_multi_modal_info = AsyncMock(return_value={})
        loop._get_mm_processor_kwargs = MagicMock(return_value={})
        loop.rollout_config = MagicMock(full_determinism=False)

        async def use_tool_then_fail(agent_data, sampling_params):
            await loop._call_tool(FakeFunctionCall("counter"), {}, agent_data)
            raise RuntimeError("generation failed")

        loop._handle_pending_state = use_tool_then_fail

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}])

        assert tool.released == ["counter-0"]


if __name__ == "__main__":
    unittest.main()
