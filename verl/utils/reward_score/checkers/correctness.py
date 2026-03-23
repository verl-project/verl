"""
Correctness checking module: Exact Match and F1 scoring.
"""

import re
import string


def normalize_answer(s: str) -> str:
    """Normalize answer text for comparison."""
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def extract_answer(solution_str: str) -> str | None:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_em(prediction: str | None, ground_truths: list[str]) -> float:
    """Exact Match score."""
    if prediction is None:
        return 0.0
    normalized_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == normalized_pred:
            return 1.0
    return 0.0


def compute_f1(prediction: str | None, ground_truths: list[str]) -> float:
    """Token-level F1 score (best across all ground truths)."""
    if prediction is None:
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        if len(common) == 0:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def compute_format_score(solution_str: str) -> float:
    """Check if model used correct <answer> format."""
    if "<answer>" in solution_str and "</answer>" in solution_str:
        return 1.0
    return 0.0
