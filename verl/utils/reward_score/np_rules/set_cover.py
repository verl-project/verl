from .common import answer_payload, minimization_score, parse_int_list


def validation(data, answer, ground_truth):
    """Validate a set-cover solution."""
    if not isinstance(data, dict) or "U" not in data or "S" not in data:
        return True, -1, "input must contain U and S"

    universe = set(data["U"])
    subsets = data["S"]
    if not isinstance(subsets, dict):
        return True, -1, "S must be a dictionary"

    try:
        payload = answer_payload(answer)
    except ValueError as exc:
        return True, -1, str(exc)

    if payload == "Impossible":
        covered_by_all = set()
        for subset in subsets.values():
            covered_by_all.update(subset)
        if covered_by_all == universe:
            return True, -1, "a cover exists, but the answer claims Impossible"
        return False, 1.0, "correctly identified as impossible"

    try:
        selected_subsets = parse_int_list(answer, allow_empty=True)
    except ValueError as exc:
        return True, -1, str(exc)

    for subset_id in selected_subsets:
        if str(subset_id) not in subsets:
            return True, -1, f"invalid subset ID: {subset_id}"

    covered = set()
    for subset_id in selected_subsets:
        covered.update(subsets[str(subset_id)])

    if covered != universe:
        missing = sorted(universe - covered)
        return True, -1, f"selected subsets do not cover U; missing={missing}"

    used_count = len(selected_subsets)
    score = minimization_score(ground_truth, used_count)
    return False, score, f"valid cover using {used_count} subsets, ground truth: {ground_truth}"
