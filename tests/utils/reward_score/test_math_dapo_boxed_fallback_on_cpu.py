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

import pytest

from verl.utils.reward_score.math_dapo import compute_score

# Final answers that the Minerva `Answer:` path already accepts after normalization.
EQUIVALENT_ANSWERS = [
    ("5", "5"),
    ("1,000", "1000"),
    (" 5 ", "5"),
    ("x = 5", "5"),
    (r"90^\circ", "90"),
    (r"\$18", "18"),
    (r"18 \text{ dollars}", "18"),
]


@pytest.mark.parametrize("answer, ground_truth", EQUIVALENT_ANSWERS)
def test_boxed_fallback_matches_minerva_path(answer, ground_truth):
    minerva = compute_score("Some reasoning.\nAnswer: " + answer, ground_truth)
    boxed = compute_score("Some reasoning, so the result is \\boxed{" + answer + "}.", ground_truth)

    assert minerva["score"] == 1.0
    assert boxed == minerva


def test_boxed_fallback_rejects_wrong_answer():
    result = compute_score(r"So the result is \boxed{1,001}.", "1000")

    assert result == {"score": -1.0, "acc": False, "pred": "1001"}


def test_strict_box_verify_still_compares_raw_text():
    result = compute_score(r"So the result is \boxed{1,000}.", "1000", strict_box_verify=True)

    assert result == {"score": -1.0, "acc": False, "pred": "1,000"}
