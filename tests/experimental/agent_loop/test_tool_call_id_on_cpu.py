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
from unittest.mock import AsyncMock, patch

import torch
from transformers import BatchFeature

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall, KimiToolParser, ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.tokenizer.deepseek import encode_messages
from verl.workers.rollout.replica import TokenOutput


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
        captured_messages: list[dict[str, Any]] = []

        async def call_tool(
            tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: SimpleNamespace
        ) -> tuple[ToolResponse, float, dict[str, Any]]:
            del tool_call, tools_kwargs, agent_data
            return ToolResponse(text='{"temperature": 26.1}'), 1.0, {}

        async def ct_merge_context_msg(
            previous_messages: list[dict[str, Any]],
            updated_messages: list[dict[str, Any]],
            runtime_token_ids: list[int],
            response_mask: list[int],
            response_logprobs: Any = None,
            *,
            tools: Any = None,
        ):
            del previous_messages, response_logprobs, tools
            captured_messages.extend(updated_messages)
            merge_result = SimpleNamespace(token_ids=list(runtime_token_ids) + [41, 42])
            return merge_result, list(response_mask) + [0, 0], None

        loop = SimpleNamespace(
            max_parallel_calls=1,
            processor=None,
            tool_parser_name="qwen3_coder",
            response_length=128,
            tool_schemas=[],
            _assert_mm_supported=lambda has_multi_modal: None,
            ct_merge_context_msg=ct_merge_context_msg,
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
            response_image_expansion=0,
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
        assert captured_messages[-1] == agent_data.messages[-1]
        assert agent_data.prompt_ids == [1, 2, 3, 41, 42]
        assert agent_data.response_mask == [0, 0]
        assert agent_data.tool_rewards == [1.0]

    async def test_deepseek_tool_loop_keeps_image_expanded_routes(self) -> None:
        async def process_multi_modal_info(messages):
            return {"images": [object()]}

        async def handle_pending(agent_data, sampling_params):
            agent_data.prompt_ids = [101, 99, 102, 11, 12, 13]
            agent_data.response_mask = [1, 1, 1]
            # One row for each token in the expanded prompt and two generated
            # tokens. The final autoregressive row was never routed.
            agent_data.routed_experts = torch.arange(8).reshape(-1, 1, 1)
            return AgentState.TERMINATED

        loop = SimpleNamespace(
            processor=SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v41")),
            rollout_config=SimpleNamespace(full_determinism=True),
            response_length=2,
            tools={},
            tool_schemas=[],
            process_multi_modal_info=process_multi_modal_info,
            _get_mm_processor_kwargs=lambda audios: {},
            _make_request_id=ToolAgentLoop._make_request_id,
            _handle_pending_state=handle_pending,
        )
        output = await ToolAgentLoop.run(loop, {}, raw_prompt=[])

        assert output.response_ids == [11, 12]
        assert output.routed_experts.flatten().tolist() == list(range(7))

    async def test_deepseek_tool_call_formats_round_trip(self) -> None:
        tool = _make_tool_schema("lookup").model_dump()
        messages = [
            {"role": "system", "content": "Look things up."},
            {"role": "user", "content": "Check Seattle."},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [{"function": {"name": "lookup", "arguments": '{"city":"Seattle","count":2}'}}],
            },
        ]
        for v41, format_name, block in ((False, "deepseek_v4", "tool_calls"), (True, "deepseek_v41", " calls")):
            with self.subTest(format=format_name):
                rendered = encode_messages(
                    messages, tools=[tool], add_generation_prompt=False, v41_reasoning_effort=v41
                )
                tag = f"<｜DSML｜{block}>"
                assert rendered.count(tag) == 3  # instruction, example and assistant call
                completion = rendered.split("<｜Assistant｜>")[-1]
                parser = ToolParser.get_tool_parser(format_name, _FakeTokenizer(completion))
                text, calls = await parser.extract_tool_calls([])
                assert text == "Checking."
                assert [call.name for call in calls] == ["lookup"]
                assert calls[0].arguments == '{"city": "Seattle", "count": 2}'

    async def test_deepseek_tool_image_uses_expanded_response_budget(self) -> None:
        async def call_tool(*args):
            return ToolResponse(image=[object()]), 0.0, {}

        async def merge_context(*args, **kwargs):
            return SimpleNamespace(token_ids=[101, 11, 98, 99, 97]), [1, 0, 0, 0], None

        async def merge_assistant(ids, tokens, mask, logprobs, **kwargs):
            return SimpleNamespace(token_ids=ids + tokens), mask + [1] * len(tokens), None

        processor = SimpleNamespace(
            image_processor=True,
            config=SimpleNamespace(
                model_type="deepseek_v41", image_token_id=99, vision_config=SimpleNamespace(downsample_ratio=1)
            ),
        )
        for budget in (5, 8):
            agent_data = AgentData([], None, None, None, {}, {}, "test", {})
            agent_data.prompt_ids = [101, 11]
            agent_data.response_mask = [1]
            agent_data.tool_calls = [FunctionCall(name="lookup", arguments="{}")]
            generate = AsyncMock(return_value=TokenOutput(token_ids=[12], stop_reason="length"))
            loop = SimpleNamespace(
                processor=processor,
                response_length=budget,
                max_parallel_calls=1,
                max_assistant_turns=0,
                max_user_turns=0,
                tool_schemas=[],
                tool_parser=SimpleNamespace(stop_token_ids=[]),
                server_manager=SimpleNamespace(generate=generate),
                _call_tool=call_tool,
                _assert_mm_supported=lambda _: None,
                ct_merge_context_msg=merge_context,
                ct_merge_assistant_token=merge_assistant,
            )
            with patch(
                "verl.experimental.agent_loop.agent_loop.build_multimodal_processor_inputs",
                return_value=BatchFeature({"image_grid_hws": torch.tensor([[1, 1]])}),
            ):
                state = await ToolAgentLoop._handle_processing_tools_state(loop, agent_data)
            if budget == 5:
                assert state is AgentState.TERMINATED
                assert agent_data.prompt_ids == [101, 11]
                assert agent_data.image_data is None and agent_data.messages == []
            else:
                assert state is AgentState.GENERATING
                assert agent_data.response_image_expansion == 3
                state = await ToolAgentLoop._handle_generating_state(loop, agent_data, {"max_new_tokens": 5})
                assert state is AgentState.TERMINATED
                assert generate.call_args.kwargs["sampling_params"] == {"max_tokens": 1}
                assert len(agent_data.response_mask) + agent_data.response_image_expansion == budget

    async def test_tool_loop_clears_prior_routes_on_abort(self) -> None:
        async def merge_assistant(ids, tokens, mask, logprobs, **kwargs):
            return SimpleNamespace(token_ids=ids), mask, None

        agent_data = AgentData([], None, None, None, {}, {}, "test", {})
        agent_data.prompt_ids = [101, 11, 12, 13]
        agent_data.response_mask = [1, 0, 0]
        agent_data.routed_experts = torch.zeros((2, 1, 1), dtype=torch.int16)
        parse_tools = AsyncMock()
        loop = SimpleNamespace(
            response_length=10,
            server_manager=SimpleNamespace(
                generate=AsyncMock(return_value=TokenOutput(token_ids=[], stop_reason="aborted"))
            ),
            tool_parser=SimpleNamespace(stop_token_ids=[], extract_tool_calls=parse_tools),
            ct_merge_assistant_token=merge_assistant,
        )
        state = await ToolAgentLoop._handle_generating_state(loop, agent_data, {})

        assert state is AgentState.TERMINATED
        assert agent_data.routed_experts is None
        parse_tools.assert_not_called()


if __name__ == "__main__":
    unittest.main()
