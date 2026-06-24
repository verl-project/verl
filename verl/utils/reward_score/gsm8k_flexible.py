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

from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score, extract_solution


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> float:
    """
    GSM8K reward with flexible answer extraction method.
    
    修复：从ground_truth中提取实际答案，而不是使用完整的解题过程字符串
    """
    # 首先从ground_truth中提取实际答案（使用strict方法匹配####格式）
    actual_ground_truth = extract_solution(ground_truth, method="strict")
    
    # 如果提取失败，尝试使用flexible方法
    if actual_ground_truth is None:
        actual_ground_truth = extract_solution(ground_truth, method="flexible")
    
    # 如果仍然提取失败，使用原始ground_truth作为 fallback
    if actual_ground_truth is None:
        actual_ground_truth = ground_truth
    
    # 使用flexible方法提取模型回答中的答案
    return gsm8k_compute_score(
        solution_str=solution_str,
        ground_truth=actual_ground_truth,
        method="flexible",
        **kwargs
    )
