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

"""CPU tests for GSM8k answer extraction around Markdown, currency and sentence punctuation."""

import pytest

from verl.utils.reward_score.gsm8k import compute_score, extract_solution


@pytest.mark.parametrize(
    "solution_str",
    [
        "2+16=18\n#### 18",
        "2+16=18\n#### **18**",
        "2+16=18\n#### *18*",
        "2+16=18\n#### __18__",
        "2+16=18\n#### `18`",
        "2+16=18\n#### $18",
        "2+16=18\n#### **$18**",
        "2+16=18\n#### 18.",
        "2+16=18\n#### **18**.",
    ],
)
def test_strict_accepts_formatted_final_answer(solution_str):
    assert extract_solution(solution_str, method="strict") == "18"
    assert compute_score(solution_str, "18") == 1.0


def test_strict_bold_final_answer_overrides_withdrawn_draft():
    # The draft "18" was withdrawn; the bold final answer is the one to score.
    solution_str = "#### 18\nWait, I made an error.\n#### **129**"
    assert extract_solution(solution_str, method="strict") == "129"
    assert compute_score(solution_str, "18") == 0.0


@pytest.mark.parametrize(
    "solution_str, expected",
    [
        ("#### 1,000", "1000"),
        ("#### -7", "-7"),
        ("#### 2.5", "2.5"),
        ("#### 2.5.", "2.5"),
        ("no marker 18", None),
    ],
)
def test_strict_keeps_existing_behaviour(solution_str, expected):
    assert extract_solution(solution_str, method="strict") == expected


@pytest.mark.parametrize("solution_str", ["So the answer is 18.", "She has 18 apples."])
def test_flexible_ignores_sentence_period(solution_str):
    assert extract_solution(solution_str, method="flexible") == "18"
    assert compute_score(solution_str, "18", method="flexible") == 1.0
