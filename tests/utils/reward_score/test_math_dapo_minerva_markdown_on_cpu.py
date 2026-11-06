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

import time

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
    "Let me solve it.\n\n`Answer:`\n\n42",
    "Let me solve it.\n\n**Answer**: 42",
    "Let me solve it.\n\nAnswer: **_42_**",
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


def test_trailing_asterisk_that_belongs_to_the_answer_is_kept():
    # A single trailing "*" can be part of the answer itself, so it is only
    # removed when it forms a matching answer wrapper (e.g. "*42*").
    correct, pred = is_correct_minerva("Answer: x*", "x*")

    assert correct is True
    assert pred == "x*"

    # The prediction must not collapse to "x" either, or it would incorrectly
    # agree with a ground truth of "x".
    correct, pred = is_correct_minerva("Answer: x*", "x")

    assert correct is False
    assert pred == "x*"


def test_leading_asterisk_is_treated_as_formatting():
    # In contrast to the trailing side, a leading "*" opens an emphasis that
    # was never closed, so it is formatting and not part of the answer.
    correct, pred = is_correct_minerva("Answer: *42", "42")

    assert correct is True
    assert pred == "42"


@pytest.mark.parametrize("solution_str", ["**Answer**: 42", "**Answer**:\n\n42", "__Answer__: 42"])
def test_markdown_marker_between_the_label_and_the_colon_is_accepted(solution_str):
    # The closing emphasis marker may sit between the label and its colon.
    correct, pred = is_correct_minerva(solution_str, "42")

    assert correct is True
    assert pred == "42"


def test_nested_emphasis_wrappers_are_removed():
    # "**_42_**" wraps the answer twice, and both wrappers are formatting.
    correct, pred = is_correct_minerva("Answer: **_42_**", "42")

    assert correct is True
    assert pred == "42"


@pytest.mark.parametrize("solution_str", ["__Answer: 42__", "**Answer: 42**", "*Answer: 42*", "`Answer: 42`"])
def test_wrapper_that_spans_the_label_and_the_answer_is_removed(solution_str):
    # The label is not part of the capture, so the closing marker of a wrapper
    # that spans the label and the answer stays inside the captured answer.
    correct, pred = is_correct_minerva(solution_str, "42")

    assert correct is True
    assert pred == "42"


@pytest.mark.parametrize("solution_str", ["__Answer__: x__", "__Answer:__ x__"])
def test_wrapper_that_is_closed_by_the_label_is_not_stripped_from_the_answer(solution_str):
    # The label closes the wrapper itself, so the trailing underscores belong to
    # the answer.
    correct, pred = is_correct_minerva(solution_str, "x__")

    assert correct is True
    assert pred == "x__"


def test_long_asterisk_run_does_not_stall_the_cleanup():
    # A long run of asterisks must not make the cleanup backtrack. The previous
    # decoration pattern needed seconds for this input.
    answer = "1" + "*" * 60 + "x"

    start = time.monotonic()
    correct, pred = is_correct_minerva(f"Answer: {answer}", answer)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0
    assert correct is True
    assert pred == answer


@pytest.mark.parametrize("answer", ["x_1", "_1", "x_", "x__"])
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


# A trailing sentence period is punctuation, not part of the answer.
PERIOD_ANSWERS = [
    "Let me solve it.\n\nAnswer: 42.",
    "Let me solve it.\n\n**Answer:** 42.",
    "Let me solve it.\n\nAnswer: **42**.",
    "Let me solve it.\n\n**Answer: 42**.",
    "Let me solve it.\n\n__Answer:__ 42.",
    "Let me solve it.\n\nAnswer: `42`.",
    "Let me solve it.\n\nAnswer: *42*.",
]


@pytest.mark.parametrize("solution_str", PERIOD_ANSWERS)
def test_minerva_answer_followed_by_a_sentence_period_is_accepted(solution_str):
    result = compute_score(solution_str, "42")

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["pred"] == "42"


@pytest.mark.parametrize("answer", ["3.14", "x_1", "x*", "\\frac{1}{2}"])
def test_a_period_inside_the_answer_is_not_stripped(answer):
    correct, pred = is_correct_minerva(f"Answer: {answer}", answer)

    assert correct is True
    assert pred == answer


def test_a_ground_truth_with_a_trailing_period_still_matches():
    correct, pred = is_correct_minerva("Answer: 42", "42.")

    assert correct is True
    assert pred == "42"
