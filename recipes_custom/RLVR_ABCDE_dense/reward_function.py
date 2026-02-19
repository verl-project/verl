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
Reward function
"""

import re

DEFAULT_CHOICES = ("A", "B", "C", "D", "E")
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
CHOICE_PATTERN = re.compile(
    r"(?:answer|option|choice)?\s*[:=]?\s*([A-Za-z])\b", re.IGNORECASE
)


def _extract_boxed_answer(text: str) -> str:
    matches = BOXED_PATTERN.findall(text)
    return matches[-1] if matches else ""


def _normalize_choice(text: str, valid_choices=DEFAULT_CHOICES) -> str:
    text = (text or "").strip().upper()
    for char in text:
        if char in valid_choices:
            return char
    return ""


def extract_choice(text: str, valid_choices=DEFAULT_CHOICES) -> str:
    """
    Extract a single-letter choice, preferring \\boxed{} values but falling back
    to phrases like "Answer: C" or the first standalone letter.
    """
    text = str(text or "")
    candidate = _normalize_choice(_extract_boxed_answer(text), valid_choices)
    if candidate:
        return candidate
    match = CHOICE_PATTERN.search(text)
    if match:
        candidate = _normalize_choice(match.group(1), valid_choices)
        if candidate:
            return candidate
    return _normalize_choice(text, valid_choices)


def char_count_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    try:
        model_choice = extract_choice(solution_str)
        gold_choice = extract_choice(ground_truth)
        return 1 if model_choice and gold_choice and model_choice == gold_choice else 0
    except Exception:
        print(ground_truth, solution_str)
        return 0
