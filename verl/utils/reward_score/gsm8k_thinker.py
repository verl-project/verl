import re

_ANSWER_CLIP_CHARS = 1000


def _get_answer_region(solution_str: str) -> str:
    """Use content after </think> if present, else fall back to the tail."""
    think_end = solution_str.rfind("</think>")
    if think_end != -1:
        region = solution_str[think_end + len("</think>"):]
    else:
        region = solution_str
    if len(region) > _ANSWER_CLIP_CHARS:
        region = region[-_ANSWER_CLIP_CHARS:]
    return region


def extract_solution(solution_str: str) -> str | None:
    region = _get_answer_region(solution_str)

    # #### <number>  (standard GSM8K format)
    m = re.findall(r"#### (\-?[0-9\.,]+)", region)
    if m:
        return m[-1].replace(",", "").replace("$", "")

    # \boxed{<number>}  (Qwen3 thinking model format)
    m = re.findall(r"\\boxed\{(\-?[0-9\.,]+)\}", region)
    if m:
        return m[-1].replace(",", "")

    # last plain number as fallback
    m = re.findall(r"(\-?[0-9\.,]+)", region)
    for candidate in reversed(m):
        if candidate not in ("", "."):
            return candidate.replace(",", "")

    return None


def compute_score(solution_str: str, ground_truth: str, score: float = 1.0, format_score: float = 0.0) -> float:
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    return score if answer == ground_truth else format_score
