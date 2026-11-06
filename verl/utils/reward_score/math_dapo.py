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

# Markdown emphasis markers that models commonly wrap around the answer label
# (e.g. "**Answer:** 42") or around the answer value (e.g. "Answer: **42**").
# Ordered longest first, so that the longest wrapper is matched.
_MARKDOWN_EMPHASIS_WRAPPERS = ("***", "___", "**", "__", "*", "_", "`")
# The characters that markdown uses for emphasis and code markers.
_MARKDOWN_MARKER_CHARS = "*`_"


def _marker_run(text: str, leading: bool) -> str:
    """Return the run of markdown marker characters at one side of the text.

    Args:
        text: The text to inspect
        leading: Whether to inspect the start instead of the end of the text

    Returns:
        The run of marker characters, empty if that side has none
    """
    if leading:
        rest = text.lstrip(_MARKDOWN_MARKER_CHARS)
        return text[: len(text) - len(rest)]
    rest = text.rstrip(_MARKDOWN_MARKER_CHARS)
    return text[len(rest) :]


def _strip_trailing_sentence_period(text: str) -> str:
    """Remove a trailing sentence period from an answer.

    Models routinely end an answer sentence with a period ("Answer: 42."), and
    that period is punctuation rather than part of the answer. It is removed
    before the markdown markers are stripped, so that a decorated answer that
    is followed by a period ("**Answer:** 42.") still reduces to the bare
    answer.

    Args:
        text: The answer text

    Returns:
        The answer without a trailing period
    """
    return text.rstrip(".")


def _strip_markdown_emphasis(text: str) -> str:
    """Remove markdown emphasis/code markers that surround an extracted answer.

    The markers are formatting, not part of the answer. Keeping them makes an
    otherwise correct answer compare unequal to the ground truth.

    A single "_" is never removed, because it is also the LaTeX subscript
    operator and answers such as "x_1" must stay untouched. A single trailing
    "*" is kept for the same reason: it can be part of the answer itself, e.g.
    "x^*".

    Args:
        text: The extracted answer text

    Returns:
        The answer without surrounding markdown emphasis markers
    """
    stripped = text.strip()

    # A wrapper that encloses the whole answer, e.g. "**42**" or "__42__".
    # Nested wrappers are removed as well, e.g. "**_42_**" -> "_42_" -> "42".
    while True:
        for wrapper in _MARKDOWN_EMPHASIS_WRAPPERS:
            if len(stripped) > 2 * len(wrapper) and stripped.startswith(wrapper) and stripped.endswith(wrapper):
                stripped = stripped[len(wrapper) : -len(wrapper)].strip()
                break
        else:
            break

    # A marker that decorates one side only, e.g. "** 42" (left over from
    # "**Answer:** 42") or "42**" (left over from "**Answer: 42**"). The run is
    # taken without a regex, because overlapping alternatives for the marker
    # lengths would backtrack exponentially on a long run of asterisks.
    # A single "_" is the LaTeX subscript operator and a single trailing "*" can
    # be part of the answer itself, so both are kept.
    leading = _marker_run(stripped, leading=True)
    if leading and leading != "_":
        stripped = stripped[len(leading) :].strip()
    trailing = _marker_run(stripped, leading=False)
    if trailing and trailing != "*" and "_" not in trailing:
        stripped = stripped[: len(stripped) - len(trailing)].strip()
    return stripped


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
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
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

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str,
    gt: str,
    gt_need_extract: bool = False,
    answer_pattern: str = (
        r"(?i)Answer(?:\*\*\*|\*\*|\*|___|__|_|`)?\s*:\s*"
        r"(?:(?:\*\*\*|\*\*|\*|___|__|_|`)(?=\s|$))?\s*([^\n]+)"
    ),
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer. The default skips an
            optional markdown emphasis or code marker placed between the label
            and the answer, so that e.g. "**Answer:**\\n\\n42" and
            "`Answer:`\\n\\n42" are parsed as "42". A marker between the label
            and its colon is skipped as well, so that e.g. "**Answer**: 42" is
            parsed as "42".

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        match = matches[-1]
        extracted_answer = match.group(1) if match.re.groups else match.group(0)
        # A wrapper can span the label and the answer, e.g. "__Answer: 42__". The
        # label is not part of the capture, so the closing marker still is. It is
        # removed when the text in front of the label opens the same marker and
        # the label itself does not close it already, e.g. "__Answer__: 42__"
        # closes the wrapper at the label. A single "_" is not a wrapper, because
        # it is the LaTeX subscript operator.
        opening = _marker_run(solution_str[: match.start()], leading=False)
        if opening and opening != "_" and extracted_answer.endswith(opening):
            label = solution_str[match.start() : match.start(1)] if match.re.groups else ""
            before_colon, _, after_colon = label.partition(":")
            closes = {
                _marker_run(before_colon.rstrip(), leading=False),
                _marker_run(after_colon.strip(), leading=False),
            }
            if opening not in closes:
                extracted_answer = extracted_answer[: -len(opening)].strip()
    else:
        extracted_answer = "[INVALID]"
    # The label may be decorated with markdown emphasis (e.g. "**Answer:** 42")
    # and the answer itself may be emphasized (e.g. "Answer: **42**").
    extracted_answer = _strip_markdown_emphasis(_strip_trailing_sentence_period(extracted_answer))
    if not extracted_answer:
        # The label carried no answer (e.g. a bare "**Answer:**" line), so fall
        # back to the `\boxed{}` answer instead of comparing against "".
        extracted_answer = "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(_strip_trailing_sentence_period(remove_boxed(last_boxed_only_string(gt))))
    else:
        gt = normalize_final_answer(_strip_trailing_sentence_period(gt))

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
