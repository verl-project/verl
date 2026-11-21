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
"""CPU tests for ``Gemma4ToolParser`` argument parsing."""

import asyncio
import json

import pytest

from verl.experimental.agent_loop.tool_parser import Gemma4ToolParser


class _TextTokenizer:
    """Decodes the "ids" as the text itself, so tests can feed model output directly."""

    pad_token = "<pad>"

    def convert_tokens_to_ids(self, token):
        return 0

    def decode(self, ids, skip_special_tokens=True):
        return ids


def _extract(text: str):
    parser = Gemma4ToolParser(_TextTokenizer())
    content, calls = asyncio.run(parser.extract_tool_calls(text))
    return content, [(call.name, json.loads(call.arguments)) for call in calls]


def test_scalar_arguments():
    text = (
        'Let me check.<|tool_call>call:get_weather{city:<|"|>Paris, FR<|"|>,days:3,'
        "ratio:0.5,metric:true,unit:null}<tool_call|>"
    )
    content, calls = _extract(text)
    assert content == "Let me check."
    assert calls == [("get_weather", {"city": "Paris, FR", "days": 3, "ratio": 0.5, "metric": True, "unit": None})]


@pytest.mark.parametrize(
    "args, expected",
    [
        ('tags:[<|"|>a<|"|>,<|"|>b, c<|"|>],ids:[1,2,3]', {"tags": ["a", "b, c"], "ids": [1, 2, 3]}),
        ('filters:{lang:<|"|>en<|"|>,year:2024},limit:5', {"filters": {"lang": "en", "year": 2024}, "limit": 5}),
        (
            'items:[{name:<|"|>x}<|"|>,qty:1},{name:<|"|>y<|"|>,qty:[2,[3]]}]',
            {"items": [{"name": "x}", "qty": 1}, {"name": "y", "qty": [2, [3]]}]},
        ),
        ("empty:{},none:[]", {"empty": {}, "none": []}),
    ],
)
def test_nested_arguments(args, expected):
    _, calls = _extract(f"<|tool_call>call:search{{{args}}}<tool_call|>")
    assert calls == [("search", expected)]


def test_function_name_with_hyphen_or_dot():
    text = (
        '<|tool_call>call:get-weather{city:<|"|>Paris<|"|>}<tool_call|>'
        '<|tool_call>call:fs.read{path:<|"|>/a<|"|>}<tool_call|>'
    )
    _, calls = _extract(text)
    assert calls == [("get-weather", {"city": "Paris"}), ("fs.read", {"path": "/a"})]


@pytest.mark.parametrize("args", ["a:[1}]", "a:[1,", "a:{b:", 'a:<|"|>open', "a:", "junk"])
def test_malformed_arguments_do_not_hang(args):
    parser = Gemma4ToolParser(_TextTokenizer())
    assert isinstance(parser._parse_arguments(args), dict)
