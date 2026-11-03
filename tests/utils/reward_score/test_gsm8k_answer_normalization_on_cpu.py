# Copyright 2025 Individual Contributor: Zh0uHX
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

"""Tests that the GSM8k scorer does not give 0 reward to correct answers because of surrounding punctuation."""

import pytest

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.gsm8k import compute_score, extract_solution


@pytest.mark.parametrize(
    "solution_str, expected",
    [
        ("#### 72", "72"),
        ("The answer is #### 72.", "72"),
        ("#### 1,000", "1000"),
        ("#### 1,000.", "1000"),
        ("#### $18", "18"),
        ("#### $1,250.", "1250"),
        ("#### -3", "-3"),
        ("#### 0.5", "0.5"),
        ("#### 18.00", "18"),
        ("#### 100", "100"),
        ("#### 7\nwait, let me redo that.\n#### 42.", "42"),
    ],
)
def test_strict_extraction(solution_str, expected):
    assert extract_solution(solution_str, method="strict") == expected


@pytest.mark.parametrize(
    "solution_str, expected",
    [
        ("Total is 72.", "72"),
        ("So she earns 1,000 dollars.", "1000"),
        ("So she pays 42, which is less than before.", "42"),
        ("The answer is 42, right, ok", "42"),
        ("Wait... the answer is 42...", "42"),
        ("It costs 0.5 dollars", "0.5"),
        ("Rory's order will cost **$29.00**.", "29"),
        ("She has 0.50 left and 300 more.", "300"),
    ],
)
def test_flexible_extraction(solution_str, expected):
    assert extract_solution(solution_str, method="flexible") == expected


@pytest.mark.parametrize("method", ["strict", "flexible"])
@pytest.mark.parametrize("solution_str", ["No numbers here, really.", "Hmm... ok.", "#### ."])
def test_punctuation_only_is_not_an_answer(solution_str, method):
    """A bare "." or "," must not be reported as an answer, or compute_score pays format_score for it."""
    assert extract_solution(solution_str, method=method) is None
    assert compute_score(solution_str, "42", method=method, format_score=0.1) == 0


def test_strict_still_requires_marker():
    assert extract_solution("The answer is 42.", method="strict") is None
    assert compute_score("The answer is 42.", "42") == 0


def test_wrong_answer_still_gets_format_score():
    assert compute_score("#### 41.", "42", format_score=0.1) == 0.1


def test_default_compute_score_accepts_trailing_period():
    """The default reward path for openai/gsm8k uses strict extraction."""
    assert default_compute_score("openai/gsm8k", "6 * 7 = 42\n#### 42.", "42") == 1.0
