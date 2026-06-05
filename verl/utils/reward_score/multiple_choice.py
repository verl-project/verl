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

import re
from typing import Any


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _completion_text(solution_str: str) -> str:
    text = _normalize_text(solution_str)
    for marker in ("<|assistant_start|>", "<|im_start|>assistant", "assistant\n"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    for marker in ("<|assistant_end|>", "<|im_end|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return re.sub(r"(?:<pad>|\s)+$", "", text, flags=re.IGNORECASE).strip()


def extract_choice_letter(solution_str: str) -> str | None:
    """Extract a multiple-choice answer letter from a model response.

    Accepted answer formats include <answer>...</answer> tags,
    JSON-ish answer fields (`"answer": "C"`), boxed answers (`\\boxed{C}`),
    explicit phrases such as `Answer: C`, `Final answer is C`,
    `Choice: C`, `Option C`, or `Letter C`, a single answer line containing
    only the letter, and a single option-style completion line such as
    `C. option text` or `C) option text`. If a prompt and completion are both
    present, known assistant markers are used to ignore prompt options.
    """
    text = _completion_text(solution_str).upper()
    patterns = [
        r"<ANSWER>\s*([A-Z])\s*</ANSWER>",
        r"['\"]ANSWER['\"]\s*:\s*['\"]?([A-Z])['\"]?",
        r"\\BOXED\{\s*([A-Z])\s*\}",
        r"\bCHOICE\s*[:\-]\s*([A-Z])\b",
        r"\bOPTION\s*[:\-]\s*([A-Z])\b",
        r"\bANSWER\s*[:\-]\s*([A-Z])\b",
        r"\bFINAL\s+ANSWER\s*[:\-]?\s*([A-Z])\b",
        r"\bFINAL\s+ANSWER\s+IS\s+([A-Z])\b",
        r"\bTHE\s+ANSWER\s+IS\s+([A-Z])\b",
        r"\b(?:OPTION|CHOICE|LETTER)\s+([A-Z])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    option_line_matches = re.findall(r"(?m)^\s*([A-Z])\s*[\).:\-]\s+\S", text)
    if len(set(option_line_matches)) == 1:
        return option_line_matches[0]
    exact_line_matches = re.findall(r"(?m)^\s*([A-Z])\s*$", text)
    return exact_line_matches[-1] if exact_line_matches else None


def compute_score(solution_str: str, ground_truth: Any, format_score: float = 0.0, score: float = 1.0) -> float:
    predicted = extract_choice_letter(solution_str)
    gold = _normalize_text(ground_truth).upper()
    if predicted is None:
        return format_score
    return score if predicted == gold else 0.0
