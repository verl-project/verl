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

"""Match each experiment dataset's answer format without relaxing DAPO training."""

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math_dapo import compute_score as compute_math_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "aime_boxed":
        result = compute_math_score(solution_str, ground_truth, strict_box_verify=True)
        # Validation treats string predictions as labels, but attempts numeric
        # aggregation for None. Use the same invalid label as Minerva without
        # changing the incorrect answer's score or accuracy.
        if result["pred"] is None:
            result["pred"] = "[INVALID]"
        return result
    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
