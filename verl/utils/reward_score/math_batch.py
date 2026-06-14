# Copyright 2025 Individual Contributor: Mert Unsal
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

from .math_reward import compute_score


def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    return [
        compute_score(solution_str, ground_truth)
        for solution_str, ground_truth in zip(solution_strs, ground_truths, strict=True)
    ]
