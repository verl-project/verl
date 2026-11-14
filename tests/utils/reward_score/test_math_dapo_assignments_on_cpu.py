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

"""CPU tests for scoring answers with multiple assignments."""

import pytest

from verl.utils.reward_score.math_dapo import compute_score


@pytest.mark.parametrize(
    "answer, ground_truth, expected_correct",
    [
        ("x=1, y=3", "x=1,y=3", True),
        ("x=2, y=3", "x=1,y=3", False),
        ("x=1, y=4", "x=1,y=3", False),
        ("y=3", "x=1,y=3", False),
        ("x=3", "3", True),
        ("x=1,234", "1234", True),
        ("x=y=3", "3", True),
        ("$x=1$, y=3", "x=1,y=3", True),
        ("$x=1$, y=4", "$x=1$, y=3", False),
        ("x=2, $y=3$", "x=1, $y=3$", False),
        ("$x=1$, $y=3$", "$x=1, y=3$", True),
        ("x=1, y_{1}=3", "x=1,y_{1}=3", True),
        ("x=2, y_{1}=3", "x=1,y_{1}=3", False),
        (r"x=1, \beta=3", r"x=1,\beta=3", True),
        (r"x=2, \beta=3", r"x=1,\beta=3", False),
        ("The value is $3$.", "3", True),
    ],
)
def test_compute_score_assignments(answer, ground_truth, expected_correct):
    result = compute_score(f"Answer: {answer}", ground_truth)
    assert result["acc"] is expected_correct
    assert result["score"] == (1.0 if expected_correct else -1.0)


@pytest.mark.parametrize(
    "answer_format",
    [
        "x=%s, y=%s",
        r"x=%s,\text{and } y=%s",
        r"x=%s \text{and} y=%s",
        "The values are $x=%s, y=%s$.",
        "$x=%s$, y=%s",
        "x=%s, $y=%s$",
        "$x=%s$, $y=%s$",
        r"$x=%s$,\text{and } $y=%s$",
        r"The values are $x=%s,\text{and } y=%s$.",
    ],
)
@pytest.mark.parametrize(
    "values, expected_score",
    [(("1", "3"), 1.0), (("2", "3"), -1.0), (("1", "4"), -1.0)],
)
def test_compute_score_assignment_format(answer_format, values, expected_score):
    answer = answer_format % values
    assert compute_score(f"Answer: {answer}", "x=1,y=3")["score"] == expected_score
    formatted_ground_truth = answer_format % ("1", "3")
    assert compute_score(f"Answer: {answer}", formatted_ground_truth)["score"] == expected_score


@pytest.mark.parametrize(
    "answer, ground_truth",
    [
        (r"x=2,\text{and } y=3", r"x=1,\text{and } y=3"),
        ("$x=1, y=3$, z=5", "$x=1, y=3$, z=4"),
        ("x=2, $y=3, z=4$", "x=1, $y=3, z=4$"),
    ],
)
def test_compute_score_compares_assignments_on_both_sides(answer, ground_truth):
    assert compute_score(f"Answer: {answer}", ground_truth)["score"] == -1.0
