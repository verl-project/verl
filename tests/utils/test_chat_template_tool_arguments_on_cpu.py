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

import copy
import json
import os

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from transformers.utils.chat_template_utils import render_jinja_template

from verl.utils.tokenizer.chat_template import apply_chat_template, tool_call_arguments_as_json

# The tool-call branch of the DeepSeek-V3.1 chat template as shipped in its
# tokenizer_config.json: ``tool['function']['arguments']`` is concatenated into the
# prompt without ``| tojson``, so a mapping raises TypeError inside Jinja. R1 and V3
# do the same inside a ```json block.
_DEEPSEEK_V31_TOOL_CALL_TEMPLATE = (
    "{{ bos_token }}"
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}{{'<｜User｜>' + message['content']}}{%- endif -%}"
    "{%- if message['role'] == 'assistant' and message['tool_calls'] -%}"
    "{{'<｜Assistant｜><think></think>'}}"
    "{%- for tool in message['tool_calls'] -%}"
    "{{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>'"
    " + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}"
    "{%- endfor -%}"
    "{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}"
    "{%- endif -%}"
    "{%- if message['role'] == 'tool' -%}"
    "{{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}"
    "{%- endif -%}"
    "{%- endfor -%}"
)


class _RecordingTokenizer:
    """Records every message list handed to ``apply_chat_template``."""

    name_or_path = "unit-test/recording"

    def __init__(self):
        self.seen = []

    def render(self, messages, add_generation_prompt):
        raise NotImplementedError

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs
    ):
        self.seen.append(messages)
        rendered = self.render(messages, add_generation_prompt)
        if tokenize:
            return [ord(char) for char in rendered]
        return rendered


class _ConcatenatingTokenizer(_RecordingTokenizer):
    """Splices tool-call arguments in by string concatenation, like DeepSeek-R1 / V3 / V3.1."""

    def render(self, messages, add_generation_prompt):
        rendered = ""
        for message in messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    function = tool_call["function"]
                    rendered += "<call>" + function["name"] + "\n```json\n" + function["arguments"] + "\n```</call>"
                continue
            rendered += f"<{message['role']}>" + message.get("content", "")
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class _JsonFilterTokenizer(_RecordingTokenizer):
    """Renders tool-call arguments through ``| tojson``, like Qwen and Llama templates."""

    def render(self, messages, add_generation_prompt):
        rendered = ""
        for message in messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    function = tool_call["function"]
                    rendered += "<call>" + json.dumps({"name": function["name"], "arguments": function["arguments"]})
                continue
            rendered += f"<{message['role']}>" + message.get("content", "")
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class _JinjaTokenizer:
    """Renders through a sandboxed Jinja environment, the way transformers does."""

    name_or_path = "unit-test/deepseek-v3.1-tool-branch"
    bos_token = "<｜begin▁of▁sentence｜>"

    def __init__(self, template=_DEEPSEEK_V31_TOOL_CALL_TEMPLATE):
        self._template = ImmutableSandboxedEnvironment().from_string(template)

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs
    ):
        rendered = self._template.render(
            messages=messages, bos_token=self.bos_token, add_generation_prompt=add_generation_prompt
        )
        if tokenize:
            return [ord(char) for char in rendered]
        return rendered


def _tool_call_history(arguments):
    return [
        {"role": "user", "content": "Weather in Seattle?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": arguments}}
            ],
        },
        {"role": "tool", "content": "rain", "tool_call_id": "c0"},
    ]


def test_tool_call_arguments_as_json_serializes_only_mappings():
    messages = _tool_call_history({"city": "Seattle"})
    original = copy.deepcopy(messages)

    converted, changed = tool_call_arguments_as_json(messages)

    assert changed
    assert converted[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Seattle"}'
    assert converted[1]["tool_calls"][0]["id"] == "c0"
    # Untouched messages are the same objects; the input is not mutated.
    assert converted[0] is messages[0]
    assert converted[2] is messages[2]
    assert messages == original


def test_tool_call_arguments_as_json_reports_nothing_to_convert():
    messages = _tool_call_history('{"city": "Seattle"}')

    converted, changed = tool_call_arguments_as_json(messages)

    assert not changed
    assert converted == messages
    assert all(after is before for after, before in zip(converted, messages, strict=True))


def test_tool_call_arguments_as_json_keeps_non_ascii_and_nesting():
    messages = _tool_call_history({"query": "天气", "ids": [1, 2], "opts": {"unit": "c"}})

    converted, _ = tool_call_arguments_as_json(messages)

    expected = '{"query": "天气", "ids": [1, 2], "opts": {"unit": "c"}}'
    assert converted[1]["tool_calls"][0]["function"]["arguments"] == expected


@pytest.mark.parametrize("tokenize", [False, True])
def test_mapping_arguments_render_through_a_concatenating_template(tokenize):
    tokenizer = _ConcatenatingTokenizer()
    messages = _tool_call_history({"city": "Seattle"})
    original = copy.deepcopy(messages)

    output = apply_chat_template(tokenizer, messages, tokenize=tokenize, add_generation_prompt=True)

    expected = (
        '<user>Weather in Seattle?<call>get_weather\n```json\n{"city": "Seattle"}\n```</call><tool>rain<assistant>'
    )
    assert output == ([ord(char) for char in expected] if tokenize else expected)
    # One failed attempt with the mapping, one retry with the JSON string.
    assert len(tokenizer.seen) == 2
    assert tokenizer.seen[0][1]["tool_calls"][0]["function"]["arguments"] == {"city": "Seattle"}
    assert tokenizer.seen[1][1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Seattle"}'
    assert messages == original


def test_string_arguments_render_in_one_attempt():
    tokenizer = _ConcatenatingTokenizer()
    messages = _tool_call_history('{"city": "Seattle"}')

    output = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False)

    assert output == '<user>Weather in Seattle?<call>get_weather\n```json\n{"city": "Seattle"}\n```</call><tool>rain'
    assert len(tokenizer.seen) == 1


def test_templates_that_render_mappings_receive_the_mapping():
    tokenizer = _JsonFilterTokenizer()
    messages = _tool_call_history({"city": "Seattle"})

    output = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False)

    assert output == (
        '<user>Weather in Seattle?<call>{"name": "get_weather", "arguments": {"city": "Seattle"}}<tool>rain'
    )
    assert len(tokenizer.seen) == 1
    assert tokenizer.seen[0] is messages


def test_deepseek_v31_tool_branch_renders_mapping_arguments_under_jinja():
    tokenizer = _JinjaTokenizer()
    messages = _tool_call_history({"city": "Seattle"})

    # The template alone rejects the mapping, with the error the DeepSeek templates raise.
    with pytest.raises(TypeError, match="can only concatenate str"):
        tokenizer.apply_chat_template(messages, tokenize=False)

    output = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False)

    assert output == (
        "<｜begin▁of▁sentence｜><｜User｜>Weather in Seattle?<｜Assistant｜><think></think>"
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Seattle"}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>rain<｜tool▁output▁end｜>"
    )


class _OfficialTemplateTokenizer:
    """Renders the DeepSeek-R1 chat template through transformers' own Jinja path.

    The template is the ``chat_template`` from the DeepSeek-R1 distill checkpoints'
    ``tokenizer_config.json``; DeepSeek-V3 shares its tool-call branch. The tool-call
    branch only runs for assistant messages whose ``content`` is ``None``, and it
    concatenates ``tool['function']['arguments']`` into the prompt.
    """

    name_or_path = "unit-test/deepseek-r1"
    bos_token = "<｜begin▁of▁sentence｜>"
    eos_token = "<｜end▁of▁sentence｜>"

    def __init__(self):
        template_path = os.path.join(os.path.dirname(__file__), "deepseek_r1_chat_template.jinja")
        with open(template_path, encoding="utf-8") as template_file:
            self.chat_template = template_file.read()

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs
    ):
        rendered, _ = render_jinja_template(
            conversations=[messages],
            tools=tools,
            chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            **kwargs,
        )
        rendered = rendered[0]
        if tokenize:
            return [ord(char) for char in rendered]
        return rendered


def test_official_deepseek_r1_template_renders_mapping_arguments():
    tokenizer = _OfficialTemplateTokenizer()
    messages = _tool_call_history({"city": "Seattle"})
    messages[1]["content"] = None

    with pytest.raises(TypeError, match="can only concatenate str"):
        tokenizer.apply_chat_template(messages, tokenize=False)

    output = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False)

    assert output == (
        "<｜begin▁of▁sentence｜><｜User｜>Weather in Seattle?"
        "<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        '```json\n{"city": "Seattle"}\n```<｜tool▁call▁end｜>'
        "<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>rain<｜tool▁output▁end｜><｜tool▁outputs▁end｜>"
    )
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Seattle"}


def test_type_errors_unrelated_to_arguments_still_propagate():
    class _BrokenTokenizer(_RecordingTokenizer):
        def render(self, messages, add_generation_prompt):
            raise TypeError("boom")

    tokenizer = _BrokenTokenizer()

    with pytest.raises(TypeError, match="boom"):
        apply_chat_template(tokenizer, _tool_call_history({"city": "Seattle"}), tokenize=False)
    with pytest.raises(TypeError, match="boom"):
        apply_chat_template(tokenizer, _tool_call_history('{"city": "Seattle"}'), tokenize=False)
