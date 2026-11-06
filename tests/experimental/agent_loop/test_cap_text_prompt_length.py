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

"""CPU-only unit tests for ``AgentLoopBase._cap_text_prompt_length``.

Regression test for #8281: the shared prompt cap used to left-truncate
(suffix-only) unconditionally, ignoring the documented ``data.truncation``
policies (``error``/``left``/``right``/``middle``). This verifies that the
cap now honors the configured policy and fails loudly on ``error``.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopBase


class _ConcreteAgentLoop(AgentLoopBase):
    """Concrete subclass used only to exercise the prompt-cap logic."""

    async def run(self, sampling_params, **kwargs):
        raise NotImplementedError


def _make_loop(prompt_length: int, truncation: str | None) -> _ConcreteAgentLoop:
    loop = _ConcreteAgentLoop.__new__(_ConcreteAgentLoop)
    loop.rollout_config = OmegaConf.create({"prompt_length": prompt_length})
    data = {"truncation": truncation} if truncation is not None else {}
    loop.data_config = OmegaConf.create(data)
    return loop


@pytest.mark.parametrize("prompt_length", [4, 5, 6])
def test_short_prompt_returned_unchanged(prompt_length: int):
    """Prompts within the cap are returned as-is under every policy."""
    prompt = list(range(prompt_length - 1))
    for truncation in ("error", "left", "right", "middle", None):
        loop = _make_loop(prompt_length, truncation)
        assert loop._cap_text_prompt_length(prompt) == prompt


def test_error_policy_raises_on_overflow():
    """Default (and explicit) ``error`` policy must raise, not silently truncate."""
    loop = _make_loop(4, "error")
    with pytest.raises(ValueError, match="exceeds rollout.prompt_length"):
        loop._cap_text_prompt_length([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_default_truncation_is_error():
    """A missing ``data.truncation`` defaults to ``error`` (matches rl_dataset.py)."""
    loop = _make_loop(4, None)
    with pytest.raises(ValueError, match="configured data.truncation='error'"):
        loop._cap_text_prompt_length([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_left_keeps_tail():
    """``left`` truncation drops the head and keeps the tail (current behavior)."""
    loop = _make_loop(4, "left")
    assert loop._cap_text_prompt_length([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == [6, 7, 8, 9]


def test_left_cap0_returns_empty_tail():
    """``left`` at cap=0 must not select the whole list via ``[-0:]``."""
    loop = _make_loop(0, "left")
    out = loop._cap_text_prompt_length(list(range(10)))
    assert len(out) == 0


def test_right_keeps_head():
    """``right`` truncation drops the tail and keeps the head."""
    loop = _make_loop(4, "right")
    assert loop._cap_text_prompt_length([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "prompt_length,expected",
    [
        (1, [0]),
        (2, [0, 9]),
        (3, [0, 8, 9]),
        (4, [0, 1, 8, 9]),
        (5, [0, 1, 7, 8, 9]),
        (6, [0, 1, 2, 7, 8, 9]),
    ],
)
def test_middle_keeps_head_and_tail(prompt_length: int, expected: list[int]):
    """``middle`` truncation keeps head+tail halves with explicit expected output.

    The head half matches ``torch_functional.postprocess_data`` (floor split, so
    the odd-cap remainder goes to the tail). At cap 1 the head half would be
    empty, so the first token is kept instead. Explicit expectations for every
    cap 1..6 prevent an incorrect interior slice from passing by re-deriving the
    expectation from the output itself.
    """
    loop = _make_loop(prompt_length, "middle")
    out = loop._cap_text_prompt_length(list(range(10)))
    assert out == expected


def test_middle_cap1_does_not_extend_prompt():
    """``middle`` at cap=1 must not select the whole list via ``[-0:]``."""
    loop = _make_loop(1, "middle")
    out = loop._cap_text_prompt_length(list(range(10)))
    assert len(out) == 1
    assert out == [0]


def test_unknown_policy_raises():
    """An unrecognized ``truncation`` value must not silently fall back to slicing."""
    loop = _make_loop(4, "banana")
    with pytest.raises(ValueError, match="data.truncation='banana'"):
        loop._cap_text_prompt_length([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
