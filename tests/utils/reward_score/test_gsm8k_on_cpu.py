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

"""Tests for the GSM8k reward function.

The response of an RL rollout does not reliably stop right after the ``#### N`` marker,
so answer extraction must not depend on the answer landing inside a fixed-size tail
window of the response.
"""

import pytest

from verl.utils.reward_score.gsm8k import _SOLUTION_CLIP_CHARS, compute_score, extract_solution


def _pad(n_chars: int) -> str:
    """Filler that contains no digit, so it can never be mistaken for an answer."""
    return ("the reasoning above confirms this result is consistent. " * (n_chars // 55 + 1))[:n_chars]


@pytest.mark.parametrize("trailing_chars", [0, 1, _SOLUTION_CLIP_CHARS - 1, _SOLUTION_CLIP_CHARS, 4000])
def test_strict_finds_answer_regardless_of_trailing_text(trailing_chars):
    """A correct answer must score 1.0 however much the model keeps writing after it."""
    solution_str = "2 + 40 = 42.\n#### 42\n" + _pad(trailing_chars)

    assert extract_solution(solution_str, method="strict") == "42"
    assert compute_score(solution_str, "42") == 1.0


@pytest.mark.parametrize("trailing_chars", [0, _SOLUTION_CLIP_CHARS, 4000])
def test_flexible_finds_last_number_regardless_of_trailing_text(trailing_chars):
    solution_str = "the answer is 42 " + _pad(trailing_chars)

    assert extract_solution(solution_str, method="flexible") == "42"


@pytest.mark.parametrize("leading_chars", [0, _SOLUTION_CLIP_CHARS, 4000])
def test_strict_takes_the_last_marker(leading_chars):
    """Only the final ``####`` marker counts, whichever window it happens to fall in."""
    solution_str = _pad(leading_chars) + "#### 7\nwait, let me redo that.\n#### 42\n" + _pad(leading_chars)

    assert extract_solution(solution_str, method="strict") == "42"


@pytest.mark.parametrize("method", ["strict", "flexible"])
def test_long_answer_straddling_the_window_boundary_is_not_truncated(method):
    """A number cut in half by the tail window must not be reported as the answer."""
    answer = "1234567890123456789"
    prefix = "#### " if method == "strict" else "the answer is "
    # Place the answer so that the tail window starts in the middle of its digits.
    tail = prefix + answer
    solution_str = _pad(_SOLUTION_CLIP_CHARS + 10 - len(answer) // 2) + tail

    assert extract_solution(solution_str, method=method) == answer


def test_no_answer_returns_none():
    for method in ("strict", "flexible"):
        assert extract_solution("no number here at all", method=method) is None
        assert extract_solution(_pad(4000), method=method) is None


def test_flexible_rejects_bare_punctuation():
    """Candidates the regex matches but that hold no digit are not answers.

    ``extract_solution`` used to leak its loop variable and return ``"."`` here, which
    ``compute_score`` then treated as a well formatted answer and paid ``format_score``
    for instead of returning 0.
    """
    solution_str = "first sentence. second sentence. third sentence."

    assert extract_solution(solution_str, method="flexible") is None
    assert compute_score(solution_str, "42", method="flexible", format_score=0.5) == 0


def test_missing_marker_scores_zero_even_when_a_number_is_present():
    """strict mode still grades formatting: no ``####`` marker means no reward."""
    solution_str = "the answer is 42"

    assert extract_solution(solution_str, method="strict") is None
    assert compute_score(solution_str, "42") == 0


def test_wrong_answer_gets_format_score():
    solution_str = "2 + 40 = 41.\n#### 41\n" + _pad(4000)

    assert compute_score(solution_str, "42", format_score=0.1) == 0.1


def test_thousands_separator_is_stripped():
    solution_str = "#### 1,234\n" + _pad(4000)

    assert extract_solution(solution_str, method="strict") == "1234"
    assert compute_score(solution_str, "1234") == 1.0
