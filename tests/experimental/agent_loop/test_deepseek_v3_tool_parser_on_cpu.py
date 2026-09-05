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
"""CPU tests for ``DeepSeekV3ToolParser``: the V3 / R1 fenced layout and the V3.1 / V3.2 plain layout."""

import unittest

from verl.experimental.agent_loop.tool_parser import DeepSeekV3ToolParser, ToolParser


class _FakeTokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, response_ids: list[int], skip_special_tokens: bool = False) -> str:
        del response_ids, skip_special_tokens
        return self.text


class TestDeepSeekV3ToolParserOnCpu(unittest.IsolatedAsyncioTestCase):
    def test_registered_under_deepseek_v3(self) -> None:
        parser = ToolParser.get_tool_parser("deepseek_v3", _FakeTokenizer(""))

        assert isinstance(parser, DeepSeekV3ToolParser)
        assert parser.stop_token_ids == []

    async def test_v3_fenced_layout(self) -> None:
        response_text = (
            "<think>\nI should look this up.\n</think>\n\nLet me check."
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Seattle", "unit": "celsius"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
            "<｜end▁of▁sentence｜>"
        )
        parser = DeepSeekV3ToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1, 2, 3])

        assert content == "<think>\nI should look this up.\n</think>\n\nLet me check."
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == '{"city": "Seattle", "unit": "celsius"}'
        assert tool_calls[0].tool_call_id is None

    async def test_v3_parallel_calls(self) -> None:
        response_text = (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Seattle"}\n```<｜tool▁call▁end｜>\n'
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Portland"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>'
        )
        parser = DeepSeekV3ToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1])

        assert content == ""
        assert [(call.name, call.arguments) for call in tool_calls] == [
            ("get_weather", '{"city": "Seattle"}'),
            ("get_weather", '{"city": "Portland"}'),
        ]

    async def test_v31_plain_layout(self) -> None:
        response_text = (
            "</think>Checking."
            '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Seattle"}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>search<｜tool▁sep｜>{"query": "forecast", "top_k": 3}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        )
        parser = DeepSeekV3ToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1])

        assert content == "</think>Checking."
        assert [(call.name, call.arguments) for call in tool_calls] == [
            ("get_weather", '{"city": "Seattle"}'),
            ("search", '{"query": "forecast", "top_k": 3}'),
        ]

    async def test_v31_plain_layout_with_a_fence_in_the_arguments(self) -> None:
        """A code-writing call: the fence belongs to the arguments, not to the layout."""
        arguments = '{"path": "a.py", "content": "```python\\nprint(1)\\n```"}'
        response_text = (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>write_file<｜tool▁sep｜>"
            f"{arguments}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        )
        parser = DeepSeekV3ToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1])

        assert content == ""
        assert [(call.name, call.arguments) for call in tool_calls] == [("write_file", arguments)]

    async def test_v3_fenced_layout_with_a_fence_in_the_arguments(self) -> None:
        """The same arguments in the fenced layout: the closing fence is the last one."""
        arguments = '{"path": "a.py", "content": "```python\\nprint(1)\\n```"}'
        response_text = (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>write_file\n"
            f"```json\n{arguments}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        )
        parser = DeepSeekV3ToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1])

        assert content == ""
        assert [(call.name, call.arguments) for call in tool_calls] == [("write_file", arguments)]

    async def test_one_section_per_call_is_read_as_parallel_calls(self) -> None:
        response_text = (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Seattle"}\n```\n<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n'
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Portland"}\n```\n<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n\n'
            "Both calls are out.<｜end▁of▁sentence｜>"
        )
        parser = DeepSeekV3ToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1])

        assert content == ""
        assert [(call.name, call.arguments) for call in tool_calls] == [
            ("get_weather", '{"city": "Seattle"}'),
            ("get_weather", '{"city": "Portland"}'),
        ]

    async def test_invalid_json_arguments_are_kept_verbatim(self) -> None:
        response_text = (
            '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": Seattle}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        parser = DeepSeekV3ToolParser(_FakeTokenizer(response_text))

        _, tool_calls = await parser.extract_tool_calls([1])

        assert len(tool_calls) == 1
        assert tool_calls[0].arguments == '{"city": Seattle}'

    async def test_plain_answer_has_no_tool_calls(self) -> None:
        response_text = "<think>\nno tool needed\n</think>\n\nSeattle is rainy.<｜end▁of▁sentence｜>"
        parser = DeepSeekV3ToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1])

        assert content == response_text
        assert tool_calls == []

    async def test_unparseable_call_is_skipped(self) -> None:
        response_text = (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>garbage without a separator<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        parser = DeepSeekV3ToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1])

        assert content == ""
        assert tool_calls == []
