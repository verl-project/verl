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

from math_verify import parse, verify


def format_reward(predict_str: str) -> float:
    """Check whether the response follows the Open-R1 format:
    exactly one <think> block followed by exactly one <answer> block.

    If the response doesn't start with <think>, we try prepending it
    (ensure_think_prefix) before matching — this is lenient toward models
    that occasionally omit the opening tag.

    <|im_end|> is intentionally omitted because verl's reward managers
    decode with skip_special_tokens=True which strips it.
    """
    # If the model omitted the opening <think> tag, prepend it so the regex can still match
    s = predict_str.strip()
    if not s.startswith("<think>"):
        s = "<think>" + s

    # All content sections reject nested/repeated tags
    pattern = (
        r"^<think>"
        r"(?:(?!</?think>|</?answer>).)*"  # think content
        r"</think>"
        r"(?:(?!</?think>|</?answer>).)*"  # separator
        r"<answer>"
        r"(?:(?!</?think>|</?answer>).)*"  # answer content
        r"</answer>$"
    )
    return 1.0 if re.fullmatch(pattern, s, re.DOTALL) else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    """Check whether the response answer is mathematically correct.

    Extracts the final answer from <answer> tags before verification,
    so intermediate equations inside <think> do not affect scoring.
    Falls back to the full string if no tags are present.
    """
    # Extract answer from <answer> tags first
    sol_match = re.search(r"<answer>(.*?)</answer>", ground_truth)
    ground_truth_answer = sol_match.group(1).strip() if sol_match else ground_truth.strip()

    content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    student_answer = content_match.group(1).strip() if content_match else predict_str.strip()

    # 1. Symbolic verification via math_verify
    try:
        answer = parse(student_answer)
        solution = parse(ground_truth_answer)
        if float(verify(answer, solution)) > 0:
            return 1.0
    except Exception:
        pass

    # 2. String-based fallback
    if student_answer == ground_truth_answer:
        return 1.0

    return 0.0


def compute_score(predict_str: str, ground_truth: str, format_score: float = 0.1) -> float:
    """Weighted combination of accuracy and format rewards.

    Anti-reward-hacking: responses with repeated </think>, <answer>,
    or </answer> tags score 0.0 for both components.
    """
    if predict_str.count("</think>") > 1 or predict_str.count("<answer>") > 1 or predict_str.count("</answer>") > 1:
        return 0.0

    return (1.0 - format_score) * acc_reward(predict_str, ground_truth) + format_score * format_reward(predict_str)
