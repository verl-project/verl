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

"""CPU tests for bold Markdown Answer: label parsing in math_dapo."""

import pytest

from verl.utils.reward_score.math_dapo import compute_score, is_correct_minerva


@pytest.mark.parametrize(
    "solution_str, gt, expected_correct, expected_pred",
    [
        ("Reasoning...\n\nAnswer: 16", "16", True, "16"),
        ("Reasoning...\n\n**Answer:** 16", "16", True, "16"),
        ("Reasoning...\n\n**answer :** 16", "16", True, "16"),
        ("Reasoning...\n\n**Answer:**\n\n16", "16", True, "16"),
        ("Reasoning...\n\n**Answer**: 16", "16", True, "16"),
        ("Reasoning...\n\n**Answer:** 15", "16", False, "15"),
        ("Reasoning...\n\n**Answer:** x_1", "x_1", True, "x_1"),
        ("Reasoning...\n\n**Answer:** x*", "x*", True, "x*"),
        ("Reasoning...\n\n**Answer:** x*", "x", False, "x*"),
        ("The answer is \\boxed{16}\n\n**Answer:**", "16", False, "[INVALID]"),
    ],
)
def test_is_correct_minerva_bold_answer_label(solution_str, gt, expected_correct, expected_pred):
    correct, pred = is_correct_minerva(solution_str, gt)
    assert correct is expected_correct
    assert pred == expected_pred

    if expected_pred != "[INVALID]":
        res = compute_score(solution_str, gt)
        assert res["acc"] is expected_correct
        assert res["score"] == (1.0 if expected_correct else -1.0)
        assert res["pred"] == expected_pred


def test_bare_bold_answer_label_falls_back_to_boxed():
    res = compute_score("The answer is \\boxed{16}\n\n**Answer:**", "16")
    assert res == {"score": 1.0, "acc": True, "pred": "16"}
