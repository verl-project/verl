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

from .common import maximization_score, parse_int_list


def validation(problem, answer, ground_truth):
    """Validate a subset-sum solution."""
    if not isinstance(problem, dict):
        return True, -1, "problem must be a dictionary"

    numbers = problem.get("numbers")
    target = problem.get("target")
    if target is None or not isinstance(numbers, dict):
        return True, -1, "problem must contain target and numbers"

    try:
        indices = parse_int_list(answer, allow_empty=True)
    except ValueError as exc:
        return True, -1, str(exc)

    if len(indices) != len(set(indices)):
        return True, -1, "answer contains duplicate indices"

    for index in indices:
        if str(index) not in numbers:
            return True, -1, f"index {index} not found in numbers"

    submitted_sum = sum(numbers[str(index)] for index in indices)
    if submitted_sum != target:
        return True, -1, f"subset sum {submitted_sum} does not match target {target}"

    subset_size = len(indices)
    score = maximization_score(subset_size, ground_truth)
    return False, score, f"valid subset of size {subset_size}, ground truth: {ground_truth}"
