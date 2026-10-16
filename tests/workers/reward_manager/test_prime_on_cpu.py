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

"""CPU tests for PrimeRewardManager dict-shaped reward scores."""

import pytest

from verl.workers.reward_manager.prime import run_reward_scoring


def _dict_score_fn(data_source, solution_str, ground_truth, extra_info=None):
    """Mirror math_dapo / rlla compute_score, which return a dict."""
    return {"score": 1.0, "acc": True, "pred": "42"}


def _float_score_fn(data_source, solution_str, ground_truth, extra_info=None):
    return 0.5


def test_run_reward_scoring_accepts_dict_scores():
    scores = run_reward_scoring(
        _dict_score_fn,
        completions=["answer"],
        references=["42"],
        tasks=["math_dapo"],
        extra_info=[None],
        num_processes=1,
    )
    assert scores == [pytest.approx(1.0)]


def test_run_reward_scoring_accepts_float_scores():
    scores = run_reward_scoring(
        _float_score_fn,
        completions=["answer"],
        references=["42"],
        tasks=["math_dapo"],
        extra_info=[None],
        num_processes=1,
    )
    assert scores == [pytest.approx(0.5)]