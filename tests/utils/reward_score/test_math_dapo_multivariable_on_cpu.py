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

"""CPU tests for multi-variable answer evaluation in math_dapo."""

import pytest

from verl.utils.reward_score.math_dapo import compute_score, is_correct_minerva, normalize_final_answer


def test_multivariable_assignment_reproduction():
    """Verify issue #8675: earlier assignments in multi-variable answers must not be ignored."""
    gold = "x=1, y=3"

    # Exact match
    res1 = compute_score("Answer: x=1, y=3", gold)
    assert res1["acc"] is True
    assert res1["score"] == 1.0
    assert res1["pred"] == "x=1,y=3"

    # First assignment differs (x=2 instead of x=1) -> must fail, not receive full credit
    res2 = compute_score("Answer: x=2, y=3", gold)
    assert res2["acc"] is False
    assert res2["score"] == -1.0
    assert res2["pred"] == "x=2,y=3"

    # Second assignment differs (y=4 instead of y=3) -> must fail
    res3 = compute_score("Answer: x=1, y=4", gold)
    assert res3["acc"] is False
    assert res3["score"] == -1.0
    assert res3["pred"] == "x=1,y=4"

    # Incomplete assignment providing only scalar '3' -> must fail
    res4 = compute_score("Answer: 3", gold)
    assert res4["acc"] is False
    assert res4["score"] == -1.0
    assert res4["pred"] == "3"


@pytest.mark.parametrize(
    "solution_str, gt, expected_correct, expected_pred",
    [
        # Multi-variable variations & commutative order
        ("Answer: y=3, x=1", "x=1, y=3", True, "x=1,y=3"),
        ("Answer: x = 1 and y = 3", "x=1, y=3", True, "x=1,y=3"),
        ("Answer: x = 1; y = 3", "x=1, y=3", True, "x=1,y=3"),
        ("Answer: x = 1, \\text{and} y = 3", "x=1, y=3", True, "x=1,y=3"),
        ("Answer: z=5, y=3, x=1", "x=1, y=3, z=5", True, "x=1,y=3,z=5"),
        ("Answer: z=5, y=4, x=1", "x=1, y=3, z=5", False, "x=1,y=4,z=5"),
        # Multi-variable math spans ($x=1$, $y=3$) and mixed delimiters ($x=1$, y=4)
        ("Answer: $x=1$, $y=4$", "$x=1$, $y=3$", False, "x=1,y=4"),
        ("Answer: $x=1$, $y=3$", "$x=1$, $y=3$", True, "x=1,y=3"),
        ("Answer: $y=3$, $x=1$", "$x=1$, $y=3$", True, "x=1,y=3"),
        ("Answer: $x=1$, y=4", "$x=1$, y=3", False, "x=1,y=4"),
        ("Answer: $x=1$, y=3", "$x=1$, y=3", True, "x=1,y=3"),
        ("Answer: x=1, $y=3$", "x=1, y=3", True, "x=1,y=3"),
        # Adjacent math spans without comma delimiter ($x=1$ $y=3$)
        ("Answer: $x=1$ $y=3$", "x=1, y=3", True, "x=1,y=3"),
        ("Answer: $x=2$ $y=3$", "x=1, y=3", False, "x=2,y=3"),
        ("Answer: $x=3$", "3", True, "3"),
        # Explanatory equations in trailing clauses ($x=1$, $y=3$ (since $2+2=4$), (using $2+2=4$))
        ("Answer: $x=1$, $y=3$ (since $2+2=4$)", "x=1, y=3", True, "x=1,y=3"),
        ("Answer: $x=1$, $y=3$ (using $2+2=4$)", "x=1, y=3", True, "x=1,y=3"),
        ("Answer: $x=1$, $y=3$ using $2+2=4$", "x=1, y=3", True, "x=1,y=3"),
        ("Answer: $x=1$, $y=3$ (with $a=1$)", "x=1, y=3", True, "x=1,y=3"),
        # Equality chains (x=1+2=3 -> 3)
        ("Answer: x=1+2=3", "3", True, "3"),
        ("Answer: a=b=c=42", "42", True, "42"),
        # Explanatory math spans after scalar answer
        ("Answer: $3$ (since $n$ is positive)", "3", True, "3"),
        # LaTeX spacing removal preserves \; without corruption
        (r"Answer: 3\;", "3", True, "3"),
        (r"Answer: 3\;", r"3\;", True, "3"),
        # Single variable scalar compatibility
        ("Answer: x = 16", "16", True, "16"),
        ("Answer: 16", "x = 16", True, "16"),
        ("Answer: x = 16", "x = 16", True, "16"),
        ("Answer: x = 15", "16", False, "15"),
        ("Answer: y = -5", "-5", True, "-5"),
        # Tuples and coordinates
        ("Answer: (x, y) = (1, 3)", "(1, 3)", True, "(1,3)"),
        ("Answer: (1, 3)", "(1, 3)", True, "(1,3)"),
        ("Answer: (1, 4)", "(1, 3)", False, "(1,4)"),
        # LaTeX boxed multi-variable
        ("Answer: \\boxed{x=1, y=3}", "x=1, y=3", True, "x=1,y=3"),
        ("Answer: x=1, y=3", "\\boxed{x=1, y=3}", True, "x=1,y=3"),
    ],
)
def test_multivariable_minerva_scoring(solution_str, gt, expected_correct, expected_pred):
    correct, pred = is_correct_minerva(solution_str, gt)
    assert correct is expected_correct
    assert pred == expected_pred

    res = compute_score(solution_str, gt)
    assert res["acc"] is expected_correct
    assert res["score"] == (1.0 if expected_correct else -1.0)
    assert res["pred"] == expected_pred


def test_normalize_final_answer_direct():
    assert normalize_final_answer("x=1, y=3") == "x=1,y=3"
    assert normalize_final_answer("y=3, x=1") == "x=1,y=3"
    assert normalize_final_answer("x = 1 and y = 3") == "x=1,y=3"
    assert normalize_final_answer("x = 16") == "16"
    assert normalize_final_answer("16") == "16"
    assert normalize_final_answer("(x, y) = (1, 3)") == "(1,3)"
