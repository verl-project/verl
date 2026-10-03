# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Reward dispatcher used for BOTH training (dapo-math-17k) and validation (AIME).

verl applies a single ``custom_reward_function`` to the train AND val reward
managers, and the manager invokes this function per-sample with the row's
``data_source``. Hardcoding AIME scoring here would silently corrupt the
dapo-math training signal, so instead we normalize custom AIME ``data_source``
tags and delegate to verl's default router, which already scores both AIME and
dapo-math through the ``math_dapo`` verifier.
"""

from verl.utils.reward_score import default_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    ds = str(data_source or "")
    # AIME val parquets may carry a tag the default router does not match
    # (it only routes ``data_source.startswith("aime")``). Normalize those so
    # validation is scored by the same math_dapo verifier used for training.
    if "aime" in ds.lower():
        ds = "aime"
    return default_compute_score(
        data_source=ds,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
