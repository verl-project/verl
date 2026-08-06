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

import json

from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)


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
