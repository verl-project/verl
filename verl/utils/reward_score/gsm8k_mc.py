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
Reward scoring function for the GSM-MC-Stage dataset (multiple choice).
"""
import re

_SOLUTION_CLIP_CHARS = 300


# We enforce the model to output the choice marker after '####'
# e.g., '...is 42. #### C' or '... #### NONE'
# The final answer may be a single uppercase letter or NONE.


def extract_choice(solution_str, method="strict"):
    """
    Extracts the final multiple-choice letter from the model's output.

    strict:
        requires format: #### C or #### NONE
    flexible:
        recovers the last valid option token near the end, even if formatting is sloppy
    """
    assert method in ["strict", "flexible"]

    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        match = re.search(r"####\s*(NONE|[A-Z])\b", solution_str)
        if match:
            return match.group(1)
        return None

    elif method == "flexible":
        candidates = re.findall(r"\b(NONE|[A-Z])\b", solution_str)
        if not candidates:
            return None
        return candidates[-1]


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    method="strict",
    format_score=0.0,
    score=1.0,
    **kwargs,
):
    """
    Scoring function for GSM-MC-Stage with strict / flexible parsing.

    Notes on API compatibility:
    - VERL reward managers call custom reward functions with:
        compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
      so we keep `extra_info` in the signature even if unused.
    - Extra keyword args may be passed by other callers; we accept `**kwargs` for robustness.
    """
    model_choice = extract_choice(solution_str=solution_str, method=method)
    format_ok = model_choice is not None

    if not format_ok:
        return {
            "score": 0.0,
            "format_ok": False,
            "pred": "",
        }

    is_correct = model_choice.strip() == ground_truth.strip()
    return {
        "score": score if is_correct else format_score,
        "format_ok": True,
        "pred": model_choice,
    }
