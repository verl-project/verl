# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


from __future__ import annotations

from typing import Any, Optional

from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score


def toolcall_shaping_reward(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    *,
    method: str = "strict",
    format_score: float = 0.1,
    score: float = 1.0,
    shaping_reward: float = 0.1,
    trigger_substring: str = "<tool_call>",
    **kwargs,
) -> float:
    """
    GSM8K reward + tool-call shaping reward (trajectory-level).
    """
    base = gsm8k_compute_score(solution_str, ground_truth, method, format_score, score)

    bonus = shaping_reward if (trigger_substring and trigger_substring in solution_str) else 0.0
    return float(base + bonus)


# Optional: keep a default name for convenience in verl config (default is compute_score) [web:59][web:65]
def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> float:
    return toolcall_shaping_reward(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
