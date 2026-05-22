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
from decimal import Decimal, InvalidOperation

_SOLUTION_CLIP_CHARS = 300
_BOXED_NUMERIC_RE = re.compile(r"\\boxed\{\s*\$?([+-]?(?:\d[\d,]*|\d*\.\d+))\s*\}")


def _normalize_numeric_token(token):
    if token is None:
        return None
    token = token.replace(",", "").replace("$", "")
    # Flexible extraction can pick up sentence-ending punctuation like "308.".
    # Strip only trailing punctuation after a digit while keeping real decimals intact.
    token = re.sub(r"(?<=\d)[\.,]+$", "", token).strip()
    if token in {"", "."}:
        return None
    return token


def _extract_boxed_numeric_token(solution_str):
    tail = solution_str[-800:] if len(solution_str) > 800 else solution_str
    matches = _BOXED_NUMERIC_RE.findall(tail)
    if not matches:
        return None
    return _normalize_numeric_token(matches[-1])


def _numeric_tokens_equal(left, right):
    if left is None or right is None:
        return False
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return left == right


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if method == "strict":
        if len(solution_str) > _SOLUTION_CLIP_CHARS:
            solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = _normalize_numeric_token(solutions[-1])
    elif method == "flexible":
        final_answer = _extract_boxed_numeric_token(solution_str)
        if final_answer is not None:
            return final_answer
        if len(solution_str) > _SOLUTION_CLIP_CHARS:
            solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    final_answer = _normalize_numeric_token(final_answer)
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
        if method == "flexible":
            correct = _numeric_tokens_equal(answer, _normalize_numeric_token(ground_truth))
        else:
            correct = answer == ground_truth
        if correct:
            return score
        else:
            return format_score
