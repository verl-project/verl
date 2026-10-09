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


@pytest.mark.parametrize("strict_box_verify", [False, True])
@pytest.mark.parametrize("boxed_length", [299, 300, 301])
def test_boxed_answer_respects_the_300_character_window(strict_box_verify, boxed_length):
    answer = "x" * (boxed_length - len(_boxed("")))
    solution = "reasoning " * 100 + _boxed(answer)

    result = compute_score(solution, answer, strict_box_verify=strict_box_verify)

    if boxed_length <= 300:
        _assert_correct(result, answer)
    else:
        assert result == {"score": -1.0, "acc": False, "pred": None if strict_box_verify else "[INVALID]"}


@pytest.mark.parametrize("strict_box_verify", [False, True])
def test_box_before_the_character_window_is_ignored(strict_box_verify):
    result = compute_score(_boxed("42") + "x" * 300, "42", strict_box_verify=strict_box_verify)

    assert result == {"score": -1.0, "acc": False, "pred": None if strict_box_verify else "[INVALID]"}


@pytest.mark.parametrize("strict_box_verify", [False, True])
def test_long_nested_boxed_answer_after_reasoning(strict_box_verify):
    answer = r"\frac{" + "x" * 151 + "}{2}"
    solution = "reasoning " * 100 + _boxed(answer)

    _assert_correct(compute_score(solution, answer, strict_box_verify=strict_box_verify), answer)


@pytest.mark.parametrize("answer, expected_score", [("42", 1.0), ("99", -1.0)])
def test_minerva_answer_takes_precedence_over_boxed_fallback(answer, expected_score):
    result = compute_score(f"Answer: {answer}\n" + _boxed("42"), "42")

    assert result == {"score": expected_score, "acc": expected_score == 1.0, "pred": answer}


@pytest.mark.parametrize("strict_box_verify", [False, True])
@pytest.mark.parametrize("pause_tokens_index", [[], [0, 1, 2], [0, 1, 2, 3, 4]])
def test_invalid_pause_token_index_count_is_rejected(strict_box_verify, pause_tokens_index):
    with pytest.raises(AssertionError):
        compute_score(_boxed("42"), "42", strict_box_verify=strict_box_verify, pause_tokens_index=pause_tokens_index)
