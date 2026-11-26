# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for ``Qwen3XMLToolParser`` parameter-value conversion.

These exercise ``_parse_xml_function_call`` directly (no tokenizer / network
needed) and in particular guard against arbitrary code execution when a
non-primitive parameter (e.g. an ``array``) is parsed from untrusted model
output.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import HermesToolParser, Qwen3XMLToolParser
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder


def _make_tool(param_name: str, param_type: str) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="list_tool",
            description="a tool used for testing",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={param_name: OpenAIFunctionPropertySchema(type=param_type)},
                required=[],
            ),
        ),
    )


def _parse_single_param(param_type: str, raw_value: str):
    parser = Qwen3XMLToolParser(tokenizer=None)
    tool = _make_tool("items", param_type)
    # The string fed to _parse_xml_function_call is exactly what _get_function_calls
    # extracts from "<function=NAME>...<parameter=KEY>VALUE</parameter>...</function>".
    function_call_str = f"list_tool><parameter=items>{raw_value}</parameter>"
    result = parser._parse_xml_function_call(function_call_str, [tool])
    return result


def test_array_literal_is_parsed():
    """A legitimate array literal is still parsed into its Python value."""
    result = _parse_single_param("array", "[1, 2, 3]")
    # arguments is a JSON string; the list literal round-trips to a JSON array.
    assert result.arguments == '{"items": [1, 2, 3]}'


def test_array_param_does_not_execute_arbitrary_code(tmp_path):
    """An ``array`` parameter must never execute code from model output.

    Regression test: the value used to be passed to ``eval()``, allowing
    arbitrary code execution. It must now be handled by ``ast.literal_eval``,
    which rejects non-literals and degenerates to the raw string.
    """
    marker = tmp_path / "pwned.txt"
    assert not marker.exists()

    payload = f'__import__("os").system("echo pwned > {marker}")'
    result = _parse_single_param("array", payload)

    # No code executed: the marker file was never created ...
    assert not marker.exists(), "ast.literal_eval must not execute arbitrary code"
    # ... and the unparseable value degenerates to the original string unchanged.
    assert json.loads(result.arguments)["items"] == payload


def test_malformed_parameter_is_skipped_but_valid_parameter_is_preserved():
    """A truncated parameter must not discard preceding valid parameters."""
    parser = Qwen3XMLToolParser(tokenizer=None)
    tool = _make_tool("items", "string")
    function_call_str = "list_tool><parameter=items>valid</parameter><parameter=truncated"

    result = parser._parse_xml_function_call(function_call_str, [tool])

    assert result is not None
    assert json.loads(result.arguments) == {"items": "valid"}


def test_malformed_function_header_returns_none():
    """A truncated function header must be ignored without raising."""
    parser = Qwen3XMLToolParser(tokenizer=None)

    assert parser._parse_xml_function_call("list_tool", []) is None


def test_missing_tools_defaults_to_empty_list():
    """Tool-call parsing must tolerate the public API's default tools value."""
    parser = Qwen3XMLToolParser(tokenizer=None)

    result = parser._parse_xml_function_call("list_tool><parameter=items>value</parameter>", None)

    assert result is not None
    assert json.loads(result.arguments) == {"items": "value"}


@pytest.mark.parametrize("parser_cls", [Qwen3XMLToolParser, HermesToolParser])
def test_call_spans_follow_sampled_token_boundaries(parser_cls):
    # The second call uses a noncanonical split of the closing tag. The last
    # one is incomplete when the generation reaches its token limit.
    pieces = [
        "prose",
        "<tool_call>",
        "1",
        "</tool_call>",
        "more",
        "<",
        "tool",
        "_call>",
        "2",
        "</",
        "tool_call",
        ">",
        "<|im_end|>",
        "<tool_call>",
        "3",
    ]
    tokenizer = SimpleNamespace(convert_ids_to_tokens=lambda ids: [pieces[i] for i in ids])
    parser = parser_cls(tokenizer=tokenizer)
    assert parser.tool_call_token_spans(list(range(len(pieces)))) == [(1, 4), (5, 12), (13, 15)]
    assert parser.tool_call_token_spans([0, 4, 12]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("skip_calls", [True, False])
async def test_call_mask_at_final_turn_preserves_prior_observations(skip_calls):
    pieces = ["<tool_call>", "argument", "</tool_call>", "answer", "<|im_end|>"]
    parser = HermesToolParser(tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda ids: [pieces[i] for i in ids]))
    # The previous tool result includes a literal unclosed tag. Its contents
    # must not affect the next independently generated assistant turn.
    data = AgentData([], None, None, None, None, {}, "request", {})
    data.prompt_ids = [50, 51, 20, 21, 0]
    data.response_mask = [1, 0, 0]
    ids = [3, 0, 1, 2, 4]
    output = SimpleNamespace(token_ids=ids, num_preempted=0, extra_fields={}, log_probs=None, routed_experts=None)
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.tool_parser = parser
    loop.server_manager = SimpleNamespace(generate=AsyncMock(return_value=output))
    loop.continuous_token_builder = ContinuousTokenBuilder(parser.tokenizer)
    loop.loop = asyncio.get_running_loop()
    loop.skip_tool_call_tokens = skip_calls
    loop.response_length = 8

    state = await loop._handle_generating_state(data, {})
    assert state is AgentState.TERMINATED
    assert data.prompt_ids == [50, 51, 20, 21, 0, *ids]
    expected_turn_mask = [1, 0, 0, 0, 1] if skip_calls else [1] * len(ids)
    assert data.response_mask == [1, 0, 0, *expected_turn_mask]
