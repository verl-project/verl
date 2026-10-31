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

_SOLUTION_CLIP_CHARS = 300


def _normalize_answer(answer):
    """Strip formatting from a matched number so it can be compared with the ground truth.

    The answer regexes also match thousands separators and sentence punctuation, e.g. ``1,000``
    or the ``72.`` in ``#### 72.``. Trailing decimal zeros are dropped as well (``28.00`` -> ``28``),
    since every GSM8k ground truth is an integer. Returns None if no digit is left, e.g. for a bare
    ``.`` or ``,``.
    """
    answer = answer.replace(",", "").rstrip(".")
    if "." in answer:
        answer = answer.rstrip("0").rstrip(".")
    if not any(ch.isdigit() for ch in answer):
        return None
    return answer


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        # an optional currency sign is allowed after the marker, e.g. "#### $18"
        solutions = re.findall("#### \\$?(\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = _normalize_answer(solutions[-1])
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        # find the last candidate that contains a digit; a bare "." or "," is punctuation, not an answer
        for candidate in reversed(answer):
            final_answer = _normalize_answer(candidate)
            if final_answer is not None:
                break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
