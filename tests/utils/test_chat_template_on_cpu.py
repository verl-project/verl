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

import pytest

from verl.utils.tokenizer.chat_template import apply_chat_template


def _text_of(message):
    content = message["content"]
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return content


class StrictTokenizer:
    """Chat template with the Qwen3.5 constraints that trigger the fallback path.

    * at least one user message is required,
    * a system message must come first,
    * only the *last* assistant message keeps its ``reasoning_content``,
      like Qwen3 / DeepSeek-R1 templates do.
    """

    def render(self, messages, add_generation_prompt=False, tools=None):
        rendered = ""
        if tools:
            rendered += f"<|tools|>{','.join(tools)}"
        for i, message in enumerate(messages):
            role = message["role"]
            rendered += f"<|{role}|>"
            if role == "assistant" and i == len(messages) - 1 and message.get("reasoning_content"):
                rendered += f"<think>{message['reasoning_content']}</think>"
            rendered += _text_of(message)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs
    ):
        if not any(m["role"] == "user" for m in messages):
            raise ValueError("chat template requires at least one user message")
        if any(m["role"] == "system" for m in messages[1:]):
            raise ValueError("System message must be at the beginning.")
        text = self.render(messages, add_generation_prompt=add_generation_prompt, tools=tools)
        if not tokenize:
            return text
        return [ord(c) for c in text]


@pytest.fixture
def tokenizer():
    return StrictTokenizer()


def _expected(tokenizer, messages, add_generation_prompt, tools=None, tokenize=False):
    text = tokenizer.render(messages, add_generation_prompt=add_generation_prompt, tools=tools)
    return [ord(c) for c in text] if tokenize else text


@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("tokenize", [False, True])
def test_single_system_message(tokenizer, add_generation_prompt, tokenize):
    """A lone system message must not raise, and must not carry a dummy user."""
    messages = [{"role": "system", "content": "you are a helpful assistant"}]
    got = apply_chat_template(tokenizer, messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    assert got == _expected(tokenizer, messages, add_generation_prompt, tokenize=tokenize)


def test_system_message_keeps_tools(tokenizer):
    messages = [{"role": "system", "content": "sys"}]
    got = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False, tools=["get_time"])
    assert got == _expected(tokenizer, messages, False, tools=["get_time"])


@pytest.mark.parametrize("tokenize", [False, True])
def test_trailing_assistant_keeps_reasoning_content(tokenizer, tokenize):
    """The dummy user must not shift the last assistant message out of last place."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "hello", "reasoning_content": "thinking hard"},
    ]
    got = apply_chat_template(tokenizer, messages, tokenize=tokenize, add_generation_prompt=False)
    expected = _expected(tokenizer, messages, False, tokenize=tokenize)
    assert got == expected
    if not tokenize:
        assert "<think>thinking hard</think>" in got


def test_no_system_message_uses_prefix_path(tokenizer):
    """Messages without a system prefix keep the original prepend-and-strip behaviour."""
    messages = [{"role": "assistant", "content": "hello", "reasoning_content": "why"}]
    got = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False)
    assert got == _expected(tokenizer, messages, False)


def test_no_fallback_when_template_accepts_messages(tokenizer):
    messages = [{"role": "user", "content": "hi"}]
    got = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=True)
    assert got == _expected(tokenizer, messages, True)
