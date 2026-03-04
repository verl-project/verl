# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py


from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed


def parse_ans(
    solution_str: str,
    total_steps: int,
) -> tuple[bool, str]:
    try:
        boxed_answer = last_boxed_only_string(solution_str[-300:])
        extracted_answer = int(remove_boxed(boxed_answer))
        if extracted_answer == -1 or 0 <= extracted_answer < total_steps:
            return extracted_answer
        else:
            return None
    except Exception:
        return None


def compute_score_fapo_genrm(
    solution_str: str,
    ground_truth: int,
    extra_info: dict,
    **kwargs,
) -> float:
    # Verify the solution
    total_steps = extra_info["total_steps"]
    extracted_answer = parse_ans(solution_str, total_steps)
    gt = "correct" if ground_truth == -1 else "incorrect"
    pred = "correct" if extracted_answer == -1 else "incorrect"
    if extracted_answer is None:
        pred = "[INVALID]"
    acc = gt == pred
    # reward = 1.0 if acc else -1.0
    if extracted_answer is None:
        reward = -1.0
    elif ground_truth == -1:
        reward = 1.0 if extracted_answer == -1 else -1.0
    else:
        # ground truth != -1
        if extracted_answer == -1:
            reward = -1.0
        else:
            # gt != -1, pred != -1
            reward = 1.0
            reward -= abs(extracted_answer - ground_truth) / total_steps

    return {
        "score": reward,
        "acc": acc,
        "pred": extracted_answer,
        "gt": ground_truth,
    }
