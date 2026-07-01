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

import unittest
from types import SimpleNamespace
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall, KimiToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class _FakeTokenizer:
    pad_token = "<pad>"

    def __init__(self, text: str):
        self.text = text

    def decode(self, response_ids: list[int], skip_special_tokens: bool = False) -> str:
        del response_ids, skip_special_tokens
        return self.text

    def convert_tokens_to_ids(self, token: str) -> int:
        return 151645 if token == "<|im_end|>" else -1


def _make_tool_schema(name: str, required: list[str] | None = None) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "tool used by CPU tests",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": required or [],
                },
            },
        }
    )


class TestToolCallIdOnCpu(unittest.IsolatedAsyncioTestCase):
    async def test_kimi_tool_parser_preserves_model_emitted_tool_call_id(self) -> None:
        raw_tool_call_id = "call-get_weather-0"
        response_text = (
            "Let me check."
            "<|tool_calls_section_begin|>"
            f"<|tool_call_begin|>{raw_tool_call_id}"
            '<|tool_call_argument_begin|>{"city":"Seattle","unit":"celsius"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            "<|im_end|>"
        )
        parser = KimiToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1, 2, 3], tools=[_make_tool_schema("get_weather")])

        assert content == "Let me check."
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == '{"city": "Seattle", "unit": "celsius"}'
        assert tool_calls[0].tool_call_id == raw_tool_call_id

    def test_build_assistant_message_preserves_tool_call_id(self) -> None:
        loop = SimpleNamespace(max_parallel_calls=2)
        agent_data = SimpleNamespace(
            tool_calls=[
                FunctionCall(name="get_weather", arguments='{"city": "Seattle"}', tool_call_id="call-get_weather-0"),
                FunctionCall(name="search", arguments='{"query": "forecast"}'),
            ]
        )

        message = ToolAgentLoop._build_assistant_message(loop, "Checking.", agent_data)

        assert message["role"] == "assistant"
        assert message["content"] == "Checking."
        assert message["tool_calls"][0] == {
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Seattle"}},
            "id": "call-get_weather-0",
        }
        assert message["tool_calls"][1] == {
            "type": "function",
            "function": {"name": "search", "arguments": {"query": "forecast"}},
        }

    async def test_processing_tools_state_adds_tool_call_id_to_tool_message(self) -> None:
        captured_add_messages: list[dict[str, Any]] = []

        async def call_tool(
            tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: SimpleNamespace
        ) -> tuple[ToolResponse, float, dict[str, Any]]:
            del tool_call, tools_kwargs, agent_data
            return ToolResponse(text='{"temperature": 26.1}'), 1.0, {}

        async def apply_chat_template(
            add_messages: list[dict[str, Any]],
            *,
            images: Any = None,
            videos: Any = None,
            remove_system_prompt: bool = False,
        ) -> list[int]:
            del images, videos, remove_system_prompt
            captured_add_messages.extend(add_messages)
            return [41, 42]

        loop = SimpleNamespace(
            max_parallel_calls=1,
            processor=None,
            enable_continuous_token=False,
            tool_parser_name="qwen3_coder",
            response_length=128,
            apply_chat_template=apply_chat_template,
            _call_tool=call_tool,
        )
        agent_data = SimpleNamespace(
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Seattle"}},
                            "id": "call-get_weather-0",
                        }
                    ],
                },
            ],
            tool_calls=[
                FunctionCall(name="get_weather", arguments='{"city": "Seattle"}', tool_call_id="call-get_weather-0")
            ],
            tools_kwargs={},
            metrics={},
            tool_rewards=[],
            prompt_ids=[1, 2, 3],
            response_mask=[],
            response_logprobs=[],
            image_data=None,
            user_turns=0,
        )

        state = await ToolAgentLoop._handle_processing_tools_state(loop, agent_data)

        assert state == AgentState.GENERATING
        assert agent_data.messages[-1] == {
            "role": "tool",
            "content": '{"temperature": 26.1}',
            "tool_call_id": "call-get_weather-0",
        }
        assert captured_add_messages == [agent_data.messages[-1]]
        assert agent_data.prompt_ids == [1, 2, 3, 41, 42]
        assert agent_data.response_mask == [0, 0]
        assert agent_data.tool_rewards == [1.0]


if __name__ == "__main__":
    unittest.main()
