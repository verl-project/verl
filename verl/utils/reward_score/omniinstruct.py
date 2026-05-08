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
"""Rule-based reward for OmniInstruct-style answer matching.

Earlier versions combined a soft lexical-overlap score with a fixed
``substring_bonus = 0.9`` whenever the normalized reference appeared inside
the normalized prediction (or vice versa). On length-bounded RL training
that interacts with two pathological dynamics:

1. ``token_f1`` and ``rouge_l_f1`` both penalize long predictions through
   their precision term, so the policy has every incentive to ramble until
   the reference shows up by accident, then collect the flat 0.9 reward.
2. Once the policy starts hitting ``data.max_response_length``, vLLM
   truncates the response *before* the model can emit the answer,
   reward collapses, and the resulting noisy gradient destabilizes
   training (we observed this on a 300-step Qwen3-Omni Thinker GSPO run:
   reward < 0.1 ↔ avg response length 1988, clip-to-cap ratio 94 percent).

This implementation keeps the original ``compute_score`` signature and the
public ``score`` / ``exact_match`` / ``substring_match`` / ``token_f1`` /
``rouge_l_f1`` keys, but tightens the substring path:

- the bonus applies only when the reference has at least
  ``min_ref_tokens_for_substring`` tokens and the prediction stays within
  ``length_factor`` of the reference;
- the bonus value defaults to ``0.5`` (down from ``0.9``) so it cannot
  dominate exact / lexical-overlap signals;
- a small additive ``format_bonus`` rewards responses that use the expected
  ``<answer>...</answer>`` schema, instead of leaving format adherence
  unmeasured;
- callers that know a response was hard-truncated by ``max_response_length``
  can pass ``extra_info={"truncated": True}`` to scale the reward down.

All knobs above can be overridden through ``extra_info`` so dataset-level
tuning does not require touching this module.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
_OPEN_ANSWER_TAG_RE = re.compile(r"<answer>", flags=re.IGNORECASE)
_PUNCT = set(string.punctuation)

DEFAULT_SUBSTRING_BONUS = 0.5
DEFAULT_FORMAT_BONUS = 0.1
DEFAULT_LENGTH_FACTOR = 3.0
DEFAULT_MIN_REF_TOKENS_FOR_SUBSTRING = 2
DEFAULT_MIN_BOUNDED_TOKENS = 8
DEFAULT_TRUNCATION_PENALTY = 0.5


def _strip_reasoning(text: str) -> str:
    text = _THINK_TAG_RE.sub(" ", text)
    answer_matches = _ANSWER_TAG_RE.findall(text)
    if answer_matches:
        text = answer_matches[-1]
    return text.strip()


def _has_answer_tag(text: str) -> bool:
    return bool(text) and _ANSWER_TAG_RE.search(text) is not None


def normalize_answer(text: str) -> str:
    text = _strip_reasoning(text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def _exact_match(prediction: str, references: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(reference) for reference in references))


def _substring_match(prediction: str, references: list[str]) -> float:
    """Plain string-containment match against the normalized text.

    Kept for backwards-compatible logging only — ``compute_score`` itself
    no longer derives reward from this metric directly.
    """
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0.0
    for reference in references:
        normalized_reference = normalize_answer(reference)
        if normalized_reference and (
            normalized_reference in normalized_prediction or normalized_prediction in normalized_reference
        ):
            return 1.0
    return 0.0


def _bounded_substring_match(
    prediction: str,
    references: list[str],
    *,
    min_ref_tokens: int,
    length_factor: float,
    min_bounded_tokens: int,
) -> tuple[float, float]:
    """Length-aware substring match.

    Returns ``(matched, max_length_ratio)``:

    - ``matched`` is ``1.0`` when at least one reference has enough tokens
      (``len(ref) >= min_ref_tokens``) AND the prediction stays within
      ``max(min_bounded_tokens, length_factor * len(ref))`` tokens.
    - ``max_length_ratio`` is the largest ``len(pred) / len(ref)`` observed
      among references that string-matched, useful for diagnostics.
    """
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0, 0.0
    pred_text = normalize_answer(prediction)

    matched = 0.0
    max_ratio = 0.0
    for reference in references:
        ref_tokens = _tokenize(reference)
        if not ref_tokens or len(ref_tokens) < min_ref_tokens:
            continue
        ref_text = normalize_answer(reference)
        if not ref_text or not (ref_text in pred_text or pred_text in ref_text):
            continue
        budget = max(min_bounded_tokens, int(length_factor * len(ref_tokens)))
        ratio = len(pred_tokens) / max(1, len(ref_tokens))
        max_ratio = max(max_ratio, ratio)
        if len(pred_tokens) <= budget:
            matched = 1.0
    return matched, max_ratio


def _token_f1(prediction: str, references: list[str]) -> float:
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0

    best = 0.0
    pred_counter = Counter(pred_tokens)
    for reference in references:
        ref_tokens = _tokenize(reference)
        if not ref_tokens:
            continue
        ref_counter = Counter(ref_tokens)
        common = sum((pred_counter & ref_counter).values())
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l_f1(prediction: str, references: list[str]) -> float:
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0

    best = 0.0
    for reference in references:
        ref_tokens = _tokenize(reference)
        if not ref_tokens:
            continue
        lcs = _lcs_length(pred_tokens, ref_tokens)
        if lcs == 0:
            continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def compute_score(solution_str, ground_truth, extra_info=None):
    """Length-aware OmniInstruct rule-based reward.

    Args:
        solution_str: model output (may include ``<think>`` and ``<answer>`` tags).
        ground_truth: a single reference string or a list of acceptable references.
        extra_info: optional dict with override knobs:

            - ``substring_bonus`` (float, default 0.5)
            - ``format_bonus`` (float, default 0.1) — added when the answer is
              wrapped in ``<answer>...</answer>`` AND there is any positive
              signal. Total score is capped to ``[0.0, 1.0]``.
            - ``length_factor`` (float, default 3.0)
            - ``min_ref_tokens_for_substring`` (int, default 2)
            - ``min_bounded_tokens`` (int, default 8)
            - ``truncated`` (bool, default False) — when the framework knows
              the response was hard-truncated by ``max_response_length``,
              set this to scale the reward down via ``truncation_penalty``.
            - ``truncation_penalty`` (float, default 0.5) — multiplier when
              ``truncated`` is True. Set to ``0.0`` for a hard zero,
              ``1.0`` to disable.

    Returns:
        Dict with at least ``score`` (in ``[0, 1]``) plus the diagnostic
        sub-metrics ``exact_match``, ``substring_match``,
        ``substring_match_bounded``, ``token_f1``, ``rouge_l_f1``,
        ``format_match``, ``pred_to_ref_length_ratio``.
    """
    extra_info = extra_info or {}
    substring_bonus_value = float(extra_info.get("substring_bonus", DEFAULT_SUBSTRING_BONUS))
    format_bonus_value = float(extra_info.get("format_bonus", DEFAULT_FORMAT_BONUS))
    length_factor = float(extra_info.get("length_factor", DEFAULT_LENGTH_FACTOR))
    min_ref_tokens = int(extra_info.get("min_ref_tokens_for_substring", DEFAULT_MIN_REF_TOKENS_FOR_SUBSTRING))
    min_bounded_tokens = int(extra_info.get("min_bounded_tokens", DEFAULT_MIN_BOUNDED_TOKENS))
    truncated = bool(extra_info.get("truncated", False))
    truncation_penalty = float(extra_info.get("truncation_penalty", DEFAULT_TRUNCATION_PENALTY))

    if isinstance(ground_truth, list):
        references = [r for r in ground_truth if r is not None and str(r).strip()]
    elif ground_truth is None:
        references = []
    else:
        references = [ground_truth] if str(ground_truth).strip() else []

    if not references:
        return {
            "score": 0.0,
            "exact_match": 0.0,
            "substring_match": 0.0,
            "substring_match_bounded": 0.0,
            "token_f1": 0.0,
            "rouge_l_f1": 0.0,
            "format_match": float(_has_answer_tag(solution_str)),
            "pred_to_ref_length_ratio": 0.0,
        }

    exact_match = _exact_match(solution_str, references)
    substring_match_raw = _substring_match(solution_str, references)
    bounded_substring, length_ratio = _bounded_substring_match(
        solution_str,
        references,
        min_ref_tokens=min_ref_tokens,
        length_factor=length_factor,
        min_bounded_tokens=min_bounded_tokens,
    )
    token_f1 = _token_f1(solution_str, references)
    rouge_l_f1 = _rouge_l_f1(solution_str, references)

    if exact_match:
        score = 1.0
    else:
        lexical_overlap = 0.5 * token_f1 + 0.5 * rouge_l_f1
        substring_contribution = substring_bonus_value if bounded_substring else 0.0
        score = max(lexical_overlap, substring_contribution)

    has_format = _has_answer_tag(solution_str)
    if has_format and score > 0.0:
        score = min(1.0, score + format_bonus_value)

    if truncated:
        score *= truncation_penalty

    score = max(0.0, min(1.0, score))

    return {
        "score": float(score),
        "exact_match": float(exact_match),
        "substring_match": float(substring_match_raw),
        "substring_match_bounded": float(bounded_substring),
        "token_f1": float(token_f1),
        "rouge_l_f1": float(rouge_l_f1),
        "format_match": float(has_format),
        "pred_to_ref_length_ratio": float(length_ratio),
    }
