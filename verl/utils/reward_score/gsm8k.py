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

# Size of the tail window that is searched before falling back to the whole response.
_SOLUTION_CLIP_CHARS = 300

_STRICT_ANSWER_RE = re.compile("#### (\\-?[0-9\\.\\,]+)")
_FLEXIBLE_ANSWER_RE = re.compile("\\-?[0-9\\.\\,]+")
_INVALID_ANSWERS = ("", ".")


def _select_last(pattern, text, is_valid):
    """Return the last match of ``pattern`` in ``text`` accepted by ``is_valid``, else None."""
    selected = None
    for match in pattern.finditer(text):
        if is_valid(match):
            selected = match
    return selected


def _find_answer_match(pattern, solution_str, is_valid):
    """Locate the answer, searching only the tail of the response when that is safe.

    Regular expression matching on very long strings can be slow, and for math problems
    the final answer is usually at the end, so a short tail window is searched first.
    The window result is only trusted when the selected match starts past the left edge
    of the window, because a match touching that edge may be the tail half of a longer
    one that the window cut in two. Otherwise the whole response is rescanned.

    Clipping the response unconditionally, which this module used to do, silently dropped
    a correct answer whenever the model kept generating for more than
    ``_SOLUTION_CLIP_CHARS`` characters after writing it.
    """
    if len(solution_str) <= _SOLUTION_CLIP_CHARS:
        return _select_last(pattern, solution_str, is_valid)

    match = _select_last(pattern, solution_str[-_SOLUTION_CLIP_CHARS:], is_valid)
    if match is not None and match.start() > 0:
        return match
    return _select_last(pattern, solution_str, is_valid)


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        # take the last solution
        match = _find_answer_match(_STRICT_ANSWER_RE, solution_str, lambda _: True)
        # no reward if there is no answer
        return None if match is None else match.group(1).replace(",", "").replace("$", "")

    # find the last number that is not '.'
    match = _find_answer_match(_FLEXIBLE_ANSWER_RE, solution_str, lambda m: m.group(0) not in _INVALID_ANSWERS)
    # no reward if there is no answer
    return None if match is None else match.group(0)


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
