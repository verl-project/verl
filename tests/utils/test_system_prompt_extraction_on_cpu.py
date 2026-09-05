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
"""CPU tests for ``extract_system_prompt_and_generation`` (verl issue #6477).

The helper derives the system-prompt prefix as ``token1[: -(len(token2) - len(token1))]``, where
``token1``/``token2`` are the token ids for one and two probe turns. When the marginal per-turn
token count (``diff``) is zero -- e.g. a template whose rendering does not grow with extra
messages -- Python's ``x[:-0]`` is ``x[:0]`` (the empty list), not "keep everything", so the whole
system prompt was silently dropped.

``MultiTurnSFTDataset`` uses the returned prefix to strip the system preamble from each turn
(``verl/utils/dataset/multiturn_sft_dataset.py``), so an empty result makes that strip a no-op and
every turn after the first keeps its own copy of the preamble.

No GPU, network, or model download is required: a self-contained Jinja2 template stands in for a
real tokenizer.
"""

import re

from jinja2 import Template

from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation

_IM_START, _IM_END, _NL = 1, 2, 3
_SPECIALS = {"<|im_start|>": _IM_START, "<|im_end|>": _IM_END, "\n": _NL}

# Ordinary ChatML template: the rendering grows with every extra turn, so the marginal per-turn
# probe (`diff`) is non-zero. This is the common case and must keep working.
_CHATML_WITH_SYSTEM = (
    "<|im_start|>system\nYou are helpful<|im_end|>\n"
    "{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# Degenerate template that only ever renders the *first* message, ignoring the rest of the list.
# One probe turn and two probe turns therefore render identically, so
# `diff == len(token2) - len(token1) == 0` -- exactly the case `x[:-0]` gets wrong.
_CHATML_FIRST_MESSAGE_ONLY = (
    "<|im_start|>system\nYou are helpful<|im_end|>\n"
    "{% set m = messages[0] %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>\n"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


class ChatMLTokenizer:
    """Deterministic, offline tokenizer mimicking a ChatML ``apply_chat_template``."""

    def __init__(self, template: str):
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


def test_zero_diff_does_not_drop_system_prompt():
    """Regression for #6477: a zero marginal turn probe must not silently empty the prompt."""
    tok = ChatMLTokenizer(_CHATML_FIRST_MESSAGE_ONLY)

    token1 = tok.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False)
    token2 = tok.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False)
    # Confirm the template really does trigger the zero-diff edge case this test targets.
    assert len(token1) == len(token2) > 0

    system_prompt, generation_prompt = extract_system_prompt_and_generation(tok)

    # The buggy `token1[:-(len(token2) - len(token1))]` == `token1[:-0]` == `[]` here. With a zero
    # marginal turn probe the entire probe render is the only information available, so it must be
    # returned as-is rather than emptied.
    assert system_prompt != []
    assert system_prompt == token1
    assert generation_prompt == [_IM_START, tok._vocab["assistant"], _NL]


def test_normal_template_unaffected():
    """Sanity check: the common (``diff > 0``) case still returns the real system prompt."""
    tok = ChatMLTokenizer(_CHATML_WITH_SYSTEM)

    system_prompt, generation_prompt = extract_system_prompt_and_generation(tok)

    # `[]` is not a valid chat-template input for every tokenizer in production, but this fake one
    # renders just the fixed system preamble, giving the ground-truth prefix to compare against.
    expected_system_prompt = tok.apply_chat_template([], add_generation_prompt=False)
    assert system_prompt == expected_system_prompt
    assert len(system_prompt) > 0
    assert generation_prompt == [_IM_START, tok._vocab["assistant"], _NL]
