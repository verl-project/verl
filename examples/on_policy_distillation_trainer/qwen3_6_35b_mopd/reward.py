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
"""Diagnostic scoring; the distillation objective does not use task rewards."""

import json
from functools import lru_cache


@lru_cache
def _scorer(task):
    import reasoning_gym

    return reasoning_gym.get_score_answer_fn(task)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    from reasoning_gym.utils import extract_answer

    label = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if label["task"] != data_source:
        raise ValueError("Teacher routing domain and scoring task disagree")
    text = (solution_str or "").replace("<|im_end|>", "")
    score = 0.0
    if "</think>" in text:
        answer = extract_answer(text.rsplit("</think>", 1)[1])
        if answer is not None:
            try:
                score = float(_scorer(data_source)(answer, label) >= 1.0)
            except (ValueError, TypeError, ArithmeticError):
                score = 0.0
    return {"score": score, "acc": score}
