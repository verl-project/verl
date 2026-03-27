# """
# Correctness checking module: Exact Match and F1 scoring.
# """

# import re
# import string


# def normalize_answer(s: str) -> str:
#     """Normalize answer text for comparison."""
#     s = s.lower().strip()
#     s = "".join(ch for ch in s if ch not in string.punctuation)
#     s = re.sub(r"\b(a|an|the)\b", " ", s)
#     s = " ".join(s.split())
#     return s


# def extract_answer(solution_str: str) -> str | None:
#     """Extract answer from <answer>...</answer> tags."""
#     match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return None


# def compute_em(prediction: str | None, ground_truths: list[str]) -> float:
#     """Exact Match score."""
#     if prediction is None:
#         return 0.0
#     normalized_pred = normalize_answer(prediction)
#     for gt in ground_truths:
#         if normalize_answer(gt) == normalized_pred:
#             return 1.0
#     return 0.0


# def compute_f1(prediction: str | None, ground_truths: list[str]) -> float:
#     """Token-level F1 score (best across all ground truths)."""
#     if prediction is None:
#         return 0.0
#     pred_tokens = normalize_answer(prediction).split()
#     if not pred_tokens:
#         return 0.0

#     best_f1 = 0.0
#     for gt in ground_truths:
#         gt_tokens = normalize_answer(gt).split()
#         if not gt_tokens:
#             continue
#         common = set(pred_tokens) & set(gt_tokens)
#         if len(common) == 0:
#             continue
#         precision = len(common) / len(pred_tokens)
#         recall = len(common) / len(gt_tokens)
#         f1 = 2 * precision * recall / (precision + recall)
#         best_f1 = max(best_f1, f1)
#     return best_f1


# def compute_format_score(solution_str: str) -> float:
#     """Check if model used correct <answer> format."""
#     if "<answer>" in solution_str and "</answer>" in solution_str:
#         return 1.0
#     return 0.0
"""
Correctness checking module: Exact Match and F1 scoring.

Changes vs original:
  - compute_format_score: now returns graded score based on answer length
    (not just 0/1 for tag presence)
  - compute_length_penalty: new helper for explicit length penalty
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
    """
    Graded format score combining:
      1. Tag presence (0.5 points): has <answer>...</answer>
      2. Answer length (0.5 points): graded by character count

    Target: answers >= 80 chars get full length score.
    This prevents the model from gaming format with a single-word answer.

    Score breakdown:
      No tag:                        0.0
      Tag but empty/very short(<20): 0.3
      Tag + short (20-79 chars):     0.3 + 0.2*(len/80)
      Tag + sufficient (>=80 chars): 0.5 + 0.5 = 1.0
    """
    has_open  = "<answer>"  in solution_str
    has_close = "</answer>" in solution_str

    if not (has_open and has_close):
        return 0.0

    answer = extract_answer(solution_str) or ""
    ans_len = len(answer.strip())

    # Tag presence: 0.5 base
    tag_score = 0.5

    # Length score: up to 0.5, target >= 80 chars
    TARGET_LEN = 80
    if ans_len < 20:
        length_score = 0.0          # effectively empty answer
    elif ans_len < TARGET_LEN:
        length_score = 0.5 * (ans_len / TARGET_LEN)
    else:
        length_score = 0.5          # full marks at 80+ chars

    return round(tag_score + length_score, 4)


def compute_length_penalty(solution_str: str,
                            min_chars: int = 50) -> float:
    """
    Hard penalty for answers that are too short.

    Returns:
        0.0  if answer is missing or shorter than min_chars  (no penalty added,
             but the caller subtracts this from the reward)
       -0.3  penalty value (negative) when answer is too short
        0.0  when answer is long enough (no penalty)

    Usage in reward:
        penalty = compute_length_penalty(solution_str)
        reward  = base_reward + penalty   # penalty is 0 or negative
    """
    answer = extract_answer(solution_str)
    if answer is None:
        return -0.5      # no tag at all — strong penalty
    if len(answer.strip()) < min_chars:
        return -0.3      # tag exists but answer too short
    return 0.0           # fine
