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
Reward function for Zhang199/TinyLLaVA-Video-R1-training-data.

  - Extract <answer> tag content from both response and label
  - Case-insensitive string match
  - Binary reward: 1.0/0.0
"""

import re

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def compute_score(predict_str: str, ground_truth: str) -> float:
    student = _extract_answer(predict_str).upper()
    if not student:
        return 0.0
    label = _extract_answer(ground_truth).upper()
    return 1.0 if student == label else 0.0
