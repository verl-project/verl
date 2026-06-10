# Copyright 2026 The VERL Team and individual contributors.
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

from typing import Any

from math_verify.grader import verify
from math_verify.parser import StringExtractionConfig, parse

CHOICE_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHOICE_EXTRACTION_TARGETS = (StringExtractionConfig(strings=CHOICE_LETTERS),)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def compute_score(
    solution_str: str, ground_truth: Any, format_score: float = 0.0, score: float = 1.0
) -> float:
    try:
        extracted_gold = parse(_as_text(ground_truth), CHOICE_EXTRACTION_TARGETS)
        extracted_pred = parse(_as_text(solution_str), CHOICE_EXTRACTION_TARGETS)
    except Exception as e:
        print(f"Error in multiple_choice string extraction: {e}")
        return format_score

    if not extracted_gold or not extracted_pred:
        return format_score
    return max(
        score if any(verify(gold, pred) for gold in extracted_gold) else 0.0
        for pred in extracted_pred
    )
