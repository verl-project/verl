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
"""CPU-only tests for ``verl/utils/chat_template.py`` system-prompt extraction.

Pins the helper contract introduced for
https://github.com/volcengine/verl/issues/6477. The previous inline
expression ``token1[: -(len(token2) - len(token1))]`` evaluated to ``[]``
via Python's ``token1[:-0] == token1[:0]`` slice semantics whenever
``len(token2) == len(token1)``, masking intent. The helper makes the
``diff <= 0`` fallback explicit and handles negative ``diff`` uniformly.
"""

from __future__ import annotations

from typing import Any

from verl.utils.chat_template import (
    _system_prompt_from_user_turns,
    extract_system_prompt_and_generation,
    initialize_system_prompt,
)

# ---------------------------------------------------------------------------
# Pure-helper tests (no tokenizer needed).
# ---------------------------------------------------------------------------


def test_system_prompt_helper_returns_prefix_when_diff_positive() -> None:
    """Happy path: ``len(token2) - len(token1)`` is the per-user-turn span,
    so peeling it off ``token1`` leaves the system-prompt prefix.
    """
    token1 = [101, 102, 200, 201]
    token2 = [101, 102, 200, 201, 200, 201]
    assert _system_prompt_from_user_turns(token1, token2) == [101, 102]


def test_system_prompt_helper_returns_empty_when_no_template_prefix() -> None:
    """Template injects no system-prompt prefix: ``token1`` is exactly one
    user turn, peeling one turn off leaves zero tokens.
    """
    token1 = [200, 201]
    token2 = [200, 201, 200, 201]
    assert _system_prompt_from_user_turns(token1, token2) == []


def test_system_prompt_helper_returns_empty_when_diff_zero() -> None:
    """Equal-length encodings for 1 vs 2 user messages — no per-turn
    boundary inferable. Pre-fix the expression ``token1[:-0]`` already
    returned ``[]``; the helper makes that fallback explicit instead of
    relying on Python slice semantics.
    """
    token1 = [101, 102, 200, 201]
    token2 = [101, 102, 200, 201]
    assert _system_prompt_from_user_turns(token1, token2) == []


def test_system_prompt_helper_returns_empty_when_diff_negative() -> None:
    """Defensive: a misbehaving template that compresses the second user
    message would have given the original expression a positive index
    slice and chopped a misleading prefix off ``token1``. Treat
    ``diff <= 0`` uniformly as "no inferable prefix".
    """
    token1 = [101, 102, 200, 201, 202]
    token2 = [101, 102, 200, 201]
    assert _system_prompt_from_user_turns(token1, token2) == []


# ---------------------------------------------------------------------------
# End-to-end with a stub tokenizer (no transformers download).
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal stub mimicking the ``apply_chat_template`` surface the
    helpers exercise. Each test pins the exact token ids returned for
    one-message vs. two-message vs. add-generation-prompt encodings.
    """

    def __init__(self, *, one_user: list[int], two_users: list[int], one_user_with_gen: list[int]) -> None:
        self._one_user = one_user
        self._two_users = two_users
        self._one_user_with_gen = one_user_with_gen

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs: Any,
    ) -> list[int]:
        del kwargs
        assert tokenize, "verl chat_template helpers always pass tokenize=True"
        if add_generation_prompt:
            return list(self._one_user_with_gen)
        if len(messages) == 1:
            return list(self._one_user)
        if len(messages) == 2:
            return list(self._two_users)
        raise AssertionError(f"unexpected message count: {len(messages)}")


def test_initialize_system_prompt_extracts_prefix() -> None:
    """Standard chat template with a system prompt prefix: peel one user-
    turn boundary off ``token1`` and return the system-prompt span.
    """
    tokenizer = _StubTokenizer(
        one_user=[101, 102, 200, 201],
        two_users=[101, 102, 200, 201, 200, 201],
        one_user_with_gen=[101, 102, 200, 201, 300],
    )
    assert initialize_system_prompt(tokenizer) == [101, 102]


def test_initialize_system_prompt_returns_empty_on_equal_length_encodings() -> None:
    """Public-API guard for the equal-length surface: pin that
    ``initialize_system_prompt`` routes through the helper rather than
    falling back into the original ``token1[:-0]`` slice expression.
    Catches a future refactor that reintroduces the inline form.
    """
    tokenizer = _StubTokenizer(
        one_user=[101, 102, 200, 201],
        two_users=[101, 102, 200, 201],
        one_user_with_gen=[101, 102, 200, 201, 300],
    )
    assert initialize_system_prompt(tokenizer) == []


def test_initialize_system_prompt_returns_empty_on_negative_diff_template() -> None:
    """Public-API guard for the ``diff < 0`` defensive arm. The original
    inline expression would give a *positive* slice index here and chop
    a misleading prefix off ``token1``; the helper returns ``[]``.
    """
    tokenizer = _StubTokenizer(
        one_user=[101, 102, 200, 201, 202],
        two_users=[101, 102, 200, 201],
        one_user_with_gen=[101, 102, 200, 201, 300],
    )
    assert initialize_system_prompt(tokenizer) == []


def test_extract_system_prompt_and_generation_round_trip() -> None:
    """``extract_system_prompt_and_generation`` returns
    ``(system_prompt, generate_prompt)``; ``generate_prompt`` is the
    ``token3 - token1`` tail.
    """
    tokenizer = _StubTokenizer(
        one_user=[101, 102, 200, 201],
        two_users=[101, 102, 200, 201, 200, 201],
        one_user_with_gen=[101, 102, 200, 201, 300, 301],
    )
    system_prompt, generate_prompt = extract_system_prompt_and_generation(tokenizer)
    assert system_prompt == [101, 102]
    assert generate_prompt == [300, 301]


def test_extract_system_prompt_and_generation_empty_on_equal_length() -> None:
    """Same equal-length guard for the second public entry point.
    Catches a refactor that re-inlines the slice in only one of the two
    call sites.
    """
    tokenizer = _StubTokenizer(
        one_user=[200, 201],
        two_users=[200, 201],
        one_user_with_gen=[200, 201, 300],
    )
    system_prompt, generate_prompt = extract_system_prompt_and_generation(tokenizer)
    assert system_prompt == []
    assert generate_prompt == [300]
