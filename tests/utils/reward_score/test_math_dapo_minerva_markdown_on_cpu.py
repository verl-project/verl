# Copyright 2025 Individual Contributor: RerankerGuo
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

"""Tests that the Minerva answer extraction in math_dapo tolerates markdown emphasis."""

import pytest

from verl.utils.reward_score.math_dapo import compute_score, is_correct_minerva

# Correct answers that are decorated with markdown emphasis must still be accepted.
MARKDOWN_ANSWERS = [
    "Let me solve it.\n\n**Answer:** 42",
    "Let me solve it.\n\n**Answer:**\n\n42",
    "Let me solve it.\n\nAnswer: **42**",
    "Let me solve it.\n\nAnswer: *42*",
    "Let me solve it.\n\n__Answer:__ 42",
    "Let me solve it.\n\n`Answer:` 42",
]


@pytest.mark.parametrize("solution_str", MARKDOWN_ANSWERS)
def test_minerva_answer_with_markdown_emphasis_is_accepted(solution_str):
    result = compute_score(solution_str, "42")

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["pred"] == "42"


@pytest.mark.parametrize("solution_str", MARKDOWN_ANSWERS)
def test_minerva_extraction_strips_markdown_emphasis(solution_str):
    correct, pred = is_correct_minerva(solution_str, "42")

    assert correct is True
    assert pred == "42"


def test_plain_answer_is_unchanged():
    result = compute_score("Let me solve it.\n\nAnswer: 42", "42")

    assert result["score"] == 1.0
    assert result["pred"] == "42"


def test_wrong_answer_with_markdown_emphasis_is_rejected():
    result = compute_score("Let me solve it.\n\n**Answer:** 41", "42")

    assert result["score"] == -1.0
    assert result["acc"] is False
    assert result["pred"] == "41"


def test_markdown_emphasis_inside_the_answer_is_kept():
    # Only the surrounding markers are formatting; the answer content is untouched.
    correct, pred = is_correct_minerva("Answer: 3*4", "3*4")

    assert correct is True
    assert pred == "3*4"


@pytest.mark.parametrize("answer", ["x_1", "_1", "x_"])
def test_latex_subscript_answer_is_not_treated_as_emphasis(answer):
    # "_" is the LaTeX subscript operator, so it must survive the extraction.
    correct, pred = is_correct_minerva(f"Answer: {answer}", answer)

    assert correct is True
    assert pred == answer


def test_bare_answer_label_falls_back_to_the_boxed_answer():
    result = compute_score("Let me solve it.\n\n**Answer:**", "42")

    # Nothing was extracted from the label, so the prediction is invalid and the
    # `\boxed{}` fallback is used (there is none here).
    assert result["score"] == -1.0
    assert result["acc"] is False
    assert result["pred"] == "[INVALID]"


def test_boxed_answer_is_unaffected():
    result = compute_score("Let me solve it.\n\nThe result is \\boxed{42}", "42")

    assert result["score"] == 1.0
    assert result["pred"] == "42"


def test_answer_label_without_value_is_rejected():
    result = compute_score("Let me solve it.\n\n**Answer:**\n\nI am not sure.", "42")

    assert result["score"] == -1.0
    assert result["acc"] is False
