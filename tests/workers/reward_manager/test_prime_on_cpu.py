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

from verl.workers.reward_manager.prime import run_reward_scoring


def _dict_score(task, completion, reference, extra_info):
    # Same shape as math_dapo.compute_score / rlla.compute_score.
    return {"score": 1.0, "acc": True, "pred": "42"}


def _float_score(task, completion, reference, extra_info):
    return 0.5


def _tuple_score(task, completion, reference, extra_info):
    # Same shape as prime_math.compute_score.
    return (True, True, "42")


def _failing_score(task, completion, reference, extra_info):
    raise RuntimeError("boom")


def _score_with(score_fn, num_rows=2):
    return run_reward_scoring(
        score_fn,
        completions=["answer"] * num_rows,
        references=["42"] * num_rows,
        tasks=["unit_test"] * num_rows,
        extra_info=[None] * num_rows,
        num_processes=1,
    )


def test_dict_result_uses_score_key():
    """dict-returning reward fns must not crash into verify()'s all-zero fallback."""
    assert _score_with(_dict_score) == [1.0, 1.0]


def test_float_result_still_works():
    assert _score_with(_float_score) == [0.5, 0.5]


def test_tuple_result_still_takes_first_element():
    assert _score_with(_tuple_score) == [1.0, 1.0]


def test_failed_task_scores_zero():
    assert _score_with(_failing_score) == [0.0, 0.0]
