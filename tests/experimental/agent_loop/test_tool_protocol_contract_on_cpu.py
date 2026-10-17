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

"""Replay protocol fixtures through real parser, message assembly, dispatch and replies.

Tokenizer decoding, Continuous Token merging, and tool side effects are test doubles.
No inference backend, network service, model weights or global tool registry is used.
"""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

_CORPUS = json.loads(Path(__file__).with_name("tool_protocol_cases.json").read_text(encoding="utf-8"))


def _schema(name):
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Offline protocol replay tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": ["object", "string", "integer"]}},
                    "required": [],
                },
            },
        }
    )


class _DecodedFixture:
    def __init__(self, text):
        self.text = text

    def decode(self, token_ids, **kwargs):
        assert token_ids == [21, 22, 23]
        return self.text


class _RecordingNativeTool:
    def __init__(self, name, calls, lifecycle, complete):
        self.name = name
        self.tool_schema = _schema(name)
        self.calls = calls
        self.lifecycle = lifecycle
        self.next_id = 0
        self.complete = complete

    async def create(self, create_kwargs):
        instance_id = f"{self.name}-{self.next_id}"
        self.next_id += 1
        self.lifecycle.append(("create", instance_id))
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        self.lifecycle.append(("execute", instance_id))
        self.calls.append((self.name, copy.deepcopy(parameters)))
        await self.complete(parameters)
        return ToolResponse(text=json.dumps(parameters, ensure_ascii=False, sort_keys=True)), 0.0, {}

    async def release(self, instance_id):
        self.lifecycle.append(("release", instance_id))


def _make_tools(kind, calls, lifecycle, complete):
    tools = {}
    for name in ("echo", "secret"):
        if kind == "native":
            tools[name] = _RecordingNativeTool(name, calls, lifecycle, complete)
        else:

            def make_function(tool_name):
                async def invoke(**parameters):
                    calls.append((tool_name, copy.deepcopy(parameters)))
                    await complete(parameters)
                    return json.dumps(parameters, ensure_ascii=False, sort_keys=True), 0.0, {}

                return invoke

            tools[name] = FunctionTool(name, make_function(name), _schema(name), is_async=True)
    return tools


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["native", "function"])
@pytest.mark.parametrize("case", _CORPUS["cases"], ids=lambda case: case["id"])
async def test_tool_protocol_reaches_dispatch_and_reply(case, kind):
    calls, lifecycle = [], []
    completions = []
    second_done = asyncio.Event()

    async def complete(parameters):
        if case["id"] == "kimi-parallel-ids":
            value = parameters["value"]
            if value == "first":
                await asyncio.wait_for(second_done.wait(), timeout=2)
            completions.append(value)
            if value == "second":
                second_done.set()

    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.tools = _make_tools(kind, calls, lifecycle, complete)
    loop.tool_schemas = [tool.tool_schema.model_dump() for tool in loop.tools.values()]
    loop.max_parallel_calls = case.get("max_parallel_calls", 8)
    loop.max_tool_response_length = 10000
    loop.tool_response_truncate_side = "left"
    loop.processor = None
    loop.response_length = 128

    # CT itself has separate tests. Preserve its inputs here; do not emulate tokenization
    # or claim that round-tripping decoded fixture text reproduces model token IDs.
    observed_contexts = []

    async def merge_context(previous_messages, messages, token_ids, response_mask, response_logprobs, *, tools):
        observed_contexts.append(copy.deepcopy(messages[len(previous_messages) :]))
        return SimpleNamespace(token_ids=list(token_ids)), list(response_mask), response_logprobs

    loop.ct_merge_context_msg = merge_context
    loop._assert_mm_supported = lambda has_mm: None
    agent_data = AgentData(
        messages=[{"role": "user", "content": "Replay the fixture."}],
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        metrics={},
        request_id="offline-protocol-fixture",
        tools_kwargs={},
    )
    agent_data.prompt_ids = [10, 21, 22, 23]
    agent_data.response_ids = [21, 22, 23]
    agent_data.response_mask = [1, 1, 1]
    agent_data.response_logprobs = [-0.1, -0.2, -0.3]
    selected = case.get("active_tools", ["echo", "secret"])
    agent_data._active_tools = {name: loop.tools[name] for name in selected}
    agent_data._active_tool_schemas = [tool.tool_schema.model_dump() for tool in agent_data._active_tools.values()]
    original_ids = list(agent_data.response_ids)

    parser = ToolParser.get_tool_parser(case["parser"], _DecodedFixture(case["text"]))
    content, parsed = await parser.extract_tool_calls(
        agent_data.response_ids, [tool.tool_schema for tool in agent_data._active_tools.values()]
    )
    actual_calls = [
        {"name": call.name, "arguments": json.loads(call.arguments), "tool_call_id": call.tool_call_id}
        for call in parsed
    ]
    assert content == case["content"]
    assert actual_calls == case["calls"]
    agent_data.tool_calls = parsed

    assistant = loop._build_assistant_message(content, agent_data)
    agent_data.messages.append(assistant)
    expected_calls = case["calls"][: loop.max_parallel_calls]
    expected_messages = []
    for call in expected_calls:
        message = {"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
        if call["tool_call_id"] is not None:
            message["id"] = call["tool_call_id"]
        expected_messages.append(message)
    assert assistant == {
        "role": "assistant",
        "content": case["content"],
        **({"tool_calls": expected_messages} if expected_messages else {}),
    }

    if parsed:
        state = await loop._handle_processing_tools_state(agent_data)
        assert state == AgentState.GENERATING
        replies = agent_data.messages[2:]
        assert len(replies) == len(expected_calls)
        assert observed_contexts == [replies]
        for call, reply in zip(expected_calls, replies, strict=True):
            assert reply["role"] == "tool"
            if call["tool_call_id"] is not None:
                assert reply["tool_call_id"] == call["tool_call_id"]
            else:
                assert "tool_call_id" not in reply
            if call["name"] in selected:
                assert json.loads(reply["content"]) == call["arguments"]
            else:
                assert "Unknown function" in reply["content"]
        assert agent_data.tool_rewards == [0.0] * len(expected_calls)
    else:
        assert observed_contexts == []

    expected_executions = [(call["name"], call["arguments"]) for call in expected_calls if call["name"] in selected]
    assert calls == expected_executions
    if case["id"] == "kimi-parallel-ids":
        assert completions == ["second", "first"]
    if kind == "native":
        created = [instance for event, instance in lifecycle if event == "create"]
        assert len(created) == len(expected_executions)
        for instance in created:
            assert [event for event, key in lifecycle if key == instance] == ["create", "execute", "release"]
    else:
        assert lifecycle == []
    assert agent_data.response_ids == original_ids
    assert agent_data.prompt_ids == [10, 21, 22, 23]
    assert agent_data.response_mask == [1, 1, 1]
    assert agent_data.response_logprobs == [-0.1, -0.2, -0.3]
