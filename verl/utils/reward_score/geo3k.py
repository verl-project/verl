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
import re

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict_str: str) -> float:
    r"""Compute a format reward based on whether the output follows the expected structure.

    Checks that the prediction contains ``<think>...</think>`` followed by a ``\boxed{}``.

    Args:
        predict_str: The model-generated prediction string.

    Returns:
        1.0 if the format matches, else 0.0.

    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    r"""Compute an accuracy reward by grading the extracted answer.

    Args:
        predict_str: The model-generated prediction string.
        ground_truth: The expected ground truth answer.
        use_boxed: If True, extract the answer from a ``\boxed{}`` wrapper.

    Returns:
        1.0 if the answer is correct, else 0.0.

    """
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    r"""Compute a weighted score combining accuracy and format rewards.

    Args:
        predict_str: The model-generated prediction string.
        ground_truth: The expected ground truth answer.
        use_boxed: If True, extract the answer from a ``\boxed{}`` wrapper.
        format_score: Weight allocated to the format reward component.

    Returns:
        A float score blending accuracy and format rewards.

    """
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )
