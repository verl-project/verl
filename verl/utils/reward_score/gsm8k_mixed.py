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
"""
Validation reward scoring for mixed GSM8K (open-ended) and GSM8K-MC datasets.
Dispatches based on data_source.
"""
from verl.utils.reward_score import gsm8k
from verl.utils.reward_score import gsm8k_mc


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Dispatch to GSM8K or GSM8K-MC scoring based on data_source."""
    if data_source == "openai/gsm8k":
        pred = gsm8k.extract_solution(solution_str=solution_str, method="strict")
        score = gsm8k.compute_score(solution_str, ground_truth)
        return {
            "score": score,
            "format_ok": pred is not None,
            "pred": pred if pred is not None else "",
        }
    if data_source in ["satoshidg/GSM-MC-Stage", "gsm8k_mc", "nocot_gsm_mc_stage"]:
        return gsm8k_mc.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
