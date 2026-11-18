# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (", and ", ", "),
    (", and", ","),
    (",and", ","),
    (" and ", ","),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    # Strip trailing explanatory clauses (e.g. "(since $2+2=4$)", "(using $2+2=4$)", "since n is positive")
    final_answer = re.sub(
        r"\s*\((?:since|because|as|where|note|for|using|by|from|via|with|given|see|per|assuming|recall|recalled)\b[^)]*\)",
        "",
        final_answer,
        flags=re.I,
    )
    final_answer = re.sub(
        r"\s*,?\s*\b(?:since|because|using|assuming|given that)\b.*", "", final_answer, flags=re.I
    )

    # Delimit adjacent math spans (e.g. "$x=1$ $y=3$" -> "$x=1$,$y=3$")
    final_answer = re.sub(r"\$\s*\$", "$,$", final_answer)

    # Remove LaTeX spacing commands like \; before converting semicolons
    final_answer = final_answer.replace(r"\;", "")
    final_answer = re.sub(r"(?<!\\);", ",", final_answer)

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # For single variable assignments or equality chains (e.g. "x = 3" or
    # "x = 1 + 2 = 3" -> "3"), strip the LHS to match scalar or tuple ground truth.
    # Multi-variable assignments (e.g. "x=1, y=3") have multiple comma-separated
    # equations and must keep all assignments intact.
    parts = [p.strip() for p in final_answer.split(",") if p.strip()]
    if sum("=" in p for p in parts) < 2 and "=" in final_answer:
        rhs = final_answer.split("=")[-1]
        if rhs.strip():
            final_answer = rhs

    # Extract and normalize LaTeX math while preserving assignments outside math spans
    math_spans = re.findall(r"\$(.*?)\$", final_answer)
    outside = re.sub(r"\$([^\$]+)\$", "", final_answer)
    if math_spans and "=" not in outside:
        assignment_spans = [s for s in math_spans if "=" in s]
        if len(assignment_spans) > 1:
            final_answer = f"${','.join(assignment_spans)}$"
        else:
            final_answer = f"${math_spans[0]}$"
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    # Normalize multi-variable assignments so order is commutative (e.g. "y=3,x=1" -> "x=1,y=3")
    if "," in final_answer:
        sub_answers = final_answer.split(",")
        if len(sub_answers) > 1 and all("=" in sub and len(sub.split("=")) == 2 for sub in sub_answers):
            sub_answers.sort()
            final_answer = ",".join(sub_answers)

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str,
    gt: str,
    gt_need_extract: bool = False,
    answer_pattern: str = r"(?i)(?:\*\*Answer(?:\*\*\s*:|\s*:\*\*)|(?<!\*)Answer\s*:)\s*([^\n]+)",
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # A bold Markdown label (**Answer:** 16) must not add its closing ** to the answer.
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    # Prefer Minerva (`Answer: ...`); fall back to `\boxed{}` when it fails.
    correct, pred = is_correct_minerva(solution_str, answer)
    if pred != "[INVALID]":
        return correct, pred

    box_correct, box_pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
    if box_pred is not None:
        return box_correct == 1, box_pred
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution (Minerva first, then `\boxed{}` fallback unless strict_box_verify)
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
