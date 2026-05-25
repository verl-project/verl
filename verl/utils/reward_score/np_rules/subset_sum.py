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
