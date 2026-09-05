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

from verl.utils.reward_score.math_dapo import compute_score


def _boxed(answer: str) -> str:
    return f"\\boxed{{{answer}}}"


def _documented_length_answer() -> tuple[str, str]:
    answer = "x" * 151
    solution = _boxed(answer)
    assert len(solution) == 159
    return solution, answer


def _assert_correct(result: dict, answer: str) -> None:
    assert result == {"score": 1.0, "acc": True, "pred": answer}


def test_documented_length_boxed_answer_in_strict_mode():
    solution, answer = _documented_length_answer()

    result = compute_score(solution, answer, strict_box_verify=True)

    _assert_correct(result, answer)


def test_documented_length_boxed_answer_in_fallback_mode():
    solution, answer = _documented_length_answer()

    result = compute_score(solution, answer)

    _assert_correct(result, answer)


def test_strict_box_uses_the_final_box():
    result = compute_score(r"\boxed{99} more reasoning \boxed{42}", "42", strict_box_verify=True)

    _assert_correct(result, "42")


def test_pause_token_indices_do_not_shorten_the_character_window():
    solution, answer = _documented_length_answer()

    result = compute_score(
        solution,
        answer,
        strict_box_verify=True,
        pause_tokens_index=[0, 1, 2, len(solution)],
    )

    _assert_correct(result, answer)


def test_strict_box_without_a_box_remains_incorrect():
    result = compute_score("reasoning only", "42", strict_box_verify=True)

    assert result == {"score": -1.0, "acc": False, "pred": None}
