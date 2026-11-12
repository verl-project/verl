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
    ],
)
def test_compute_score_assignments(answer, ground_truth, expected_correct):
    result = compute_score(f"Answer: {answer}", ground_truth)
    assert result["acc"] is expected_correct
    assert result["score"] == (1.0 if expected_correct else -1.0)
