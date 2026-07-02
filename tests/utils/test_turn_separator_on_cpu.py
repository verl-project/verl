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
"""CPU tests for ``initialize_turn_separator`` and the multi-turn agent-loop token boundary.

These exercise the chat-template utility directly with a self-contained ChatML tokenizer, so no
GPU, network, or model download is required. They reproduce the incremental-encoding token drop
described in verl issue #6501 (multi-turn agent loop) and verify the separator restores parity
with ``apply_chat_template`` of the full conversation.
"""

import re

from jinja2 import Template

from verl.utils.tokenizer.chat_template import initialize_system_prompt, initialize_turn_separator

# Qwen-style ChatML: every turn renders as ``<|im_start|>{role}\n{content}<|im_end|>\n`` with a
# trailing ``\n`` turn separator that generation never emits (the model stops at ``<|im_end|>``).
_CHATML = (
    "{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
# Same structure but without the trailing per-turn separator (no ``\n`` after ``<|im_end|>``).
_CHATML_NO_SEP = (
    "{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
# Thinking model (e.g. Qwen3): assistant turns are wrapped in ``<think></think>`` scaffolding. The
# turn separator is still just ``\n`` -- deriving it from an assistant turn would wrongly capture
# the scaffold, which is why the helper probes user turns instead.
_CHATML_THINK = (
    "{% for m in messages %}<|im_start|>{{m['role']}}\n"
    "{% if m['role'] == 'assistant' %}<think>\n\n</think>\n\n{% endif %}"
    "{{m['content']}}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

_IM_START, _IM_END, _NL = 1, 2, 3
_SPECIALS = {"<|im_start|>": _IM_START, "<|im_end|>": _IM_END, "\n": _NL}


class ChatMLTokenizer:
    """Deterministic, offline tokenizer mimicking a ChatML ``apply_chat_template``."""

    eos_token_id = _IM_END

    def __init__(self, template: str = _CHATML):
        self._template = Template(template)
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for piece in re.split(r"(<\|im_start\|>|<\|im_end\|>|\n)", text):
            if piece == "":
                continue
            if piece in _SPECIALS:
                ids.append(_SPECIALS[piece])
            else:
                for word in piece.split(" "):
                    if word:
                        ids.append(self._vocab.setdefault(word, len(self._vocab) + 100))
        return ids

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, tools=None, **kwargs):
        text = self._template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        return self.encode(text) if tokenize else text


def _rollout_ids(tokenizer, system_prompt, turn_separator, turns):
    """Reproduce ToolAgentLoop's incremental token construction.

    ``turns`` is a list of ``(assistant_text, tool_content)`` pairs. Each assistant turn is the
    model output ending at ``<|im_end|>`` (never the template's trailing separator); each tool turn
    is rendered in isolation with the system prompt stripped, then the separator is restored.
    """
    user = {"role": "user", "content": "what is the weather in Paris?"}
    ids = tokenizer.apply_chat_template([user], add_generation_prompt=True)
    for assistant_text, tool_content in turns:
        ids = ids + tokenizer.encode(assistant_text) + [_IM_END]
        tool_iso = tokenizer.apply_chat_template(
            [{"role": "tool", "content": tool_content}], add_generation_prompt=True
        )[len(system_prompt) :]
        ids = ids + (turn_separator + tool_iso)
    return ids


def _reference_ids(tokenizer, turns):
    user = {"role": "user", "content": "what is the weather in Paris?"}
    messages = [user]
    for assistant_text, tool_content in turns:
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "tool", "content": tool_content})
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def test_turn_separator_is_the_newline_token():
    tok = ChatMLTokenizer()
    assert initialize_turn_separator(tok) == [_NL]


def test_no_separator_template_returns_empty():
    tok = ChatMLTokenizer(_CHATML_NO_SEP)
    assert initialize_turn_separator(tok) == []


def test_separator_with_list_valued_eos_and_multi_token_close():
    """Guard for list-valued ``eos_token_id`` (e.g. Llama 3) with a multi-token turn close.

    Each turn closes with two special tokens before the ``\\n`` separator, so the naive
    ``suffix[1:]`` fallback would leave the second close token in the separator. Only splitting
    after the last matching eos id recovers ``[\\n]`` -- and that split is skipped unless a
    list-valued ``eos_token_id`` is handled, which is exactly the reviewed case.
    """
    end, im_end, nl = 10, 11, 12
    specials = {"<|s|>": 9, "<|end|>": end, "<|im_end|>": im_end, "\n": nl}
    template = Template(
        "{% for m in messages %}<|s|>{{m['role']}}\n{{m['content']}}<|end|><|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|s|>assistant\n{% endif %}"
    )

    class MultiCloseTokenizer:
        eos_token_id = [end, im_end]  # list rather than a single int

        def __init__(self):
            self._vocab: dict[str, int] = {}

        def encode(self, text, add_special_tokens=False):
            ids = []
            for piece in re.split(r"(<\|s\|>|<\|end\|>|<\|im_end\|>|\n)", text):
                if piece == "":
                    continue
                if piece in specials:
                    ids.append(specials[piece])
                else:
                    for word in piece.split(" "):
                        if word:
                            ids.append(self._vocab.setdefault(word, len(self._vocab) + 100))
            return ids

        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, tools=None, **kwargs):
            text = template.render(messages=messages, add_generation_prompt=add_generation_prompt)
            return self.encode(text) if tokenize else text

    assert initialize_turn_separator(MultiCloseTokenizer()) == [nl]


def test_separator_ignores_assistant_reasoning_scaffold():
    """Regression guard for the Qwen3 ``<think>`` gotcha.

    A thinking template injects ``<think></think>`` scaffolding into assistant turns. Deriving the
    separator from an assistant probe (the original, buggy approach) captures that scaffold; the
    helper probes user turns instead and must return just the newline separator.
    """
    tok = ChatMLTokenizer(_CHATML_THINK)

    # The template really does inject scaffolding, so a naive assistant-turn probe is wrong: this is
    # exactly what the first implementation computed, and it is not the separator.
    base = tok.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True)
    closed = tok.apply_chat_template(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}], add_generation_prompt=False
    )
    assistant_probe_result = closed[len(base) :][1:]
    assert assistant_probe_result != [_NL]

    # The real helper avoids the scaffold and recovers the true separator.
    assert initialize_turn_separator(tok) == [_NL]


def test_incremental_encoding_matches_full_render_single_turn():
    tok = ChatMLTokenizer()
    system_prompt = initialize_system_prompt(tok)
    separator = initialize_turn_separator(tok)
    turns = [('<tool_call>\n{"name": "get_weather"}\n</tool_call>', "sunny, 25C")]

    # Without the separator the incremental sequence drops one token per turn boundary.
    buggy = _rollout_ids(tok, system_prompt, [], turns)
    reference = _reference_ids(tok, turns)
    assert buggy != reference
    assert len(buggy) == len(reference) - 1

    # Restoring the separator recovers exact parity with the full-conversation render.
    fixed = _rollout_ids(tok, system_prompt, separator, turns)
    assert fixed == reference


def test_incremental_encoding_matches_full_render_multi_turn():
    tok = ChatMLTokenizer()
    system_prompt = initialize_system_prompt(tok)
    separator = initialize_turn_separator(tok)
    turns = [
        ('<tool_call>\n{"name": "get_weather"}\n</tool_call>', "sunny, 25C"),
        ('<tool_call>\n{"name": "get_time"}\n</tool_call>', "14:00"),
        ('<tool_call>\n{"name": "get_news"}\n</tool_call>', "all quiet"),
    ]
    fixed = _rollout_ids(tok, system_prompt, separator, turns)
    reference = _reference_ids(tok, turns)
    assert fixed == reference
    # The buggy path drops exactly one token per tool turn boundary.
    buggy = _rollout_ids(tok, system_prompt, [], turns)
    assert len(reference) - len(buggy) == len(turns)


def test_separator_with_default_system_prompt_template():
    template = (
        "<|im_start|>system\nYou are helpful<|im_end|>\n"
        "{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    tok = ChatMLTokenizer(template)
    system_prompt = initialize_system_prompt(tok)
    separator = initialize_turn_separator(tok)
    assert separator == [_NL]
    turns = [("assistant answer", "tool result")]
    fixed = _rollout_ids(tok, system_prompt, separator, turns)
    assert fixed == _reference_ids(tok, turns)
