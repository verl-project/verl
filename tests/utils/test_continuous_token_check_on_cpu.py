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

import json
import logging
from types import SimpleNamespace

import pytest

from verl.utils.tokenizer.continuous_token_check import (
    _simulate_generation,
    _stop_token_ids,
    check_continuous_token_builder,
    verify_continuous_token_builder,
)
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

_LOGGER = "verl.utils.tokenizer.continuous_token_check"


class _ChatMLTokenizer:
    """Character-level tokenizer with ChatML special tokens and a configurable template.

    ``separator`` follows every ``<|im_end|>``. Assistant reasoning renders as a
    ``<think>`` block; with ``thinking`` the last assistant turn also gets an empty one,
    so earlier turns re-render once later messages are appended (like Qwen3).
    ``fail_on_tools`` makes every render that passes tools raise.
    """

    name_or_path = "unit-test/chatml"
    eos_token_id = 1001
    _SPECIAL = {"<|im_start|>": 1000, "<|im_end|>": 1001}

    def __init__(
        self,
        *,
        separator="\n",
        thinking=False,
        fail_on_tools=False,
        generation_prompt="<|im_start|>assistant\n",
    ):
        self.separator = separator
        self.thinking = thinking
        self.fail_on_tools = fail_on_tools
        self.generation_prompt = generation_prompt

    def convert_tokens_to_ids(self, token):
        return self._SPECIAL.get(token)

    def encode(self, text, add_special_tokens=False):
        ids, index = [], 0
        while index < len(text):
            special = next((token for token in self._SPECIAL if text.startswith(token, index)), None)
            if special is None:
                ids.append(ord(text[index]))
                index += 1
            else:
                ids.append(self._SPECIAL[special])
                index += len(special)
        return ids

    def decode(self, ids):
        names = {token_id: token for token, token_id in self._SPECIAL.items()}
        return "".join(names.get(token_id, chr(token_id)) for token_id in ids)

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs
    ):
        if tools and self.fail_on_tools:
            raise ValueError("template does not support tools")
        parts = []
        for index, message in enumerate(messages):
            content = message.get("content", "")
            if message["role"] == "assistant":
                reasoning = message.get("reasoning_content")
                if reasoning or (self.thinking and index == len(messages) - 1):
                    content = f"<think>{reasoning or ''}</think>{content}"
                for call in message.get("tool_calls", []):
                    content += "<tool_call>" + json.dumps(call["function"]) + "</tool_call>"
            parts.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>{self.separator}")
        if add_generation_prompt:
            parts.append(self.generation_prompt)
        text = "".join(parts)
        return self.encode(text) if tokenize else text


def _verify(tokenizer, *, mode="strict", hf_model_type="qwen2"):
    builder = create_continuous_token_builder(tokenizer, hf_model_type=hf_model_type)
    verify_continuous_token_builder(builder, mode=mode, hf_model_type=hf_model_type, generation_config=None)


def _statuses(tokenizer, hf_model_type):
    builder = create_continuous_token_builder(tokenizer, hf_model_type=hf_model_type)
    return {result.status for result in check_continuous_token_builder(builder, stop_token_ids={1001})}


def test_matching_builder_passes(caplog):
    tokenizer = _ChatMLTokenizer()
    assert _statuses(tokenizer, "qwen2") == {"match"}
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _verify(tokenizer)
    assert "at 8 multi-turn boundaries (8 exact matches, 0 could not be rendered)" in caplog.text


def test_dropped_separator_raises_in_strict_mode():
    tokenizer = _ChatMLTokenizer()
    # ``llama`` has no registered builder, so the default builder drops the "\n" after <|im_end|>.
    assert _statuses(tokenizer, "llama") == {"boundary"}
    with pytest.raises(ValueError, match=r"ContinuousTokenBuilder \(config.json model_type='llama'\)") as exc_info:
        _verify(tokenizer, hf_model_type="llama")
    message = str(exc_info.value)
    assert "tool.merge_at3.tool" in message
    assert r"<|im_end|>\n<|im_start|>tool" in message
    assert "tokenization_sanity_check_mode=disable" in message


def test_ignore_strippable_tolerates_whitespace_only_mismatch(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _verify(_ChatMLTokenizer(), mode="ignore_strippable", hf_model_type="llama")
    assert "tolerated by ignore_strippable" in caplog.text


def test_ignore_strippable_still_raises_on_token_mismatch():
    with pytest.raises(ValueError, match="does not reproduce the chat template"):
        _verify(_ChatMLTokenizer(separator="#"), mode="ignore_strippable", hf_model_type="llama")


def test_history_rewrite_only_warns(caplog):
    tokenizer = _ChatMLTokenizer(thinking=True)
    # Without reasoning, the empty <think> of the last assistant turn disappears once the
    # turn is no longer last; with reasoning the template keeps it.
    assert _statuses(tokenizer, "qwen3") == {"history", "match"}
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _verify(tokenizer, hf_model_type="qwen3")
    assert "re-renders earlier turns" in caplog.text
    assert "<think>" in caplog.text


def test_unrenderable_template_is_skipped(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _verify(_ChatMLTokenizer(fail_on_tools=True))
    assert "Could not verify Continuous Token builder" in caplog.text
    assert "template does not support tools" in caplog.text


def test_unlocatable_stop_token_is_skipped():
    # The generation prompt opens a reasoning block that assistant turns without reasoning
    # never render, so only the reasoning variants can be simulated.
    tokenizer = _ChatMLTokenizer(generation_prompt="<|im_start|>assistant\n<think>")
    builder = create_continuous_token_builder(tokenizer, hf_model_type="qwen3")
    results = check_continuous_token_builder(builder, stop_token_ids={1001})
    skipped = [result for result in results if result.status == "skipped"]
    assert {result.case.split(".")[0] for result in skipped} == {"tool", "user"}
    assert all("could not locate the assistant stop token" in result.detail for result in skipped)
    assert {result.status for result in results if result not in skipped} == {"match"}


def test_disable_skips_the_check():
    # The builder is never touched when the check is disabled.
    verify_continuous_token_builder(object(), mode="disable", hf_model_type=None, generation_config=None)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="tokenization_sanity_check_mode must be one of"):
        _verify(_ChatMLTokenizer(), mode="warn")


@pytest.mark.parametrize(
    ("prompt_ids", "turn_ids", "expected_ids", "runtime_ids"),
    [
        # The stop token closes the assistant turn; the template newline after it is not generated.
        ([1, 2], [1, 2, 3, 9, 10], [1, 2, 3, 9, 10, 4], [1, 2, 3, 9]),
        # GLM style: the turn has no stop token, the model stops on the next message's role token.
        ([1, 2], [1, 2, 3], [1, 2, 3, 9, 4], [1, 2, 3, 9]),
    ],
)
def test_simulate_generation(prompt_ids, turn_ids, expected_ids, runtime_ids):
    assert _simulate_generation(prompt_ids, turn_ids, expected_ids, {9}) == runtime_ids


@pytest.mark.parametrize(
    ("prompt_ids", "turn_ids", "expected_ids"),
    [
        # A stop token inside the prompt is not the assistant's stop token.
        ([9, 2], [9, 2, 3], [9, 2, 3, 4]),
        # The render ends where the single-pass render ends, so there is no next token.
        ([1, 2], [1, 2, 3], [1, 2, 3]),
        # The generation prompt is not a prefix of the rendered assistant turn.
        ([1, 5], [1, 2, 3, 9], [1, 2, 3, 9, 4]),
    ],
)
def test_simulate_generation_without_stop_token_raises(prompt_ids, turn_ids, expected_ids):
    with pytest.raises(ValueError, match="could not locate the assistant stop token"):
        _simulate_generation(prompt_ids, turn_ids, expected_ids, {9})


def test_stop_token_ids_merges_generation_config_and_tokenizer():
    tokenizer = SimpleNamespace(eos_token_id=7)
    assert _stop_token_ids(tokenizer, SimpleNamespace(eos_token_id=[5, 6, None])) == {5, 6, 7}
    assert _stop_token_ids(SimpleNamespace(eos_token_id=(3, 4)), None) == {3, 4}
    assert _stop_token_ids(SimpleNamespace(), None) == set()
