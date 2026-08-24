# Copyright 2025 Meituan Ltd. and/or its affiliates
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

from verl.experimental.fully_async_policy.detach_utils import summarize_trajectory_staleness


def test_trajectory_staleness_summary_reports_batch_mean():
    stale_count, mean_age, max_age = summarize_trajectory_staleness(
        trajectory_min_param_versions=[5, 4, 4, 5],
        current_param_version=5,
    )

    assert stale_count == 2
    assert mean_age == 0.5
    assert max_age == 1


def test_trajectory_staleness_mean_weights_partial_rollout_tokens():
    stale_count, mean_age, max_age = summarize_trajectory_staleness(
        trajectory_min_param_versions=[4],
        current_param_version=5,
        trajectory_global_step_token_counts=[{5: 50, 4: 50}],
    )

    assert stale_count == 1
    assert mean_age == 0.5
    assert max_age == 1


def test_trajectory_staleness_empty_token_counts_fall_back_to_worst_age():
    stale_count, mean_age, max_age = summarize_trajectory_staleness(
        trajectory_min_param_versions=[3, 5],
        current_param_version=5,
        trajectory_global_step_token_counts=[{}, None],
    )

    assert stale_count == 1
    assert mean_age == 1.0
    assert max_age == 2


def test_trajectory_staleness_token_counts_length_mismatch_raises():
    with pytest.raises(ValueError, match="one entry per trajectory"):
        summarize_trajectory_staleness(
            trajectory_min_param_versions=[5, 5],
            current_param_version=5,
            trajectory_global_step_token_counts=[{5: 10}],
        )
