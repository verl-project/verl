from .common import maximization_score, parse_int_list


def validation(data, answer, ground_truth):
    """Validate a knapsack solution."""
    if not isinstance(data, dict):
        return True, -1, "input must be a dictionary"
    if "capacity" not in data or "items" not in data:
        return True, -1, "input must contain capacity and items"

    try:
        capacity = int(data["capacity"])
    except (TypeError, ValueError):
        return True, -1, "capacity must be an integer"
    if capacity < 0:
        return True, -1, "capacity must be non-negative"

    items = data["items"]
    if not isinstance(items, dict):
        return True, -1, "items must be a dictionary"
    items = {str(item_id): item for item_id, item in items.items()}

    try:
        chosen = parse_int_list(answer, allow_empty=True)
    except ValueError as exc:
        return True, -1, str(exc)

    if len(chosen) != len(set(chosen)):
        return True, -1, "duplicate item IDs in answer"

    total_weight = 0
    total_value = 0
    for item_id in chosen:
        item = items.get(str(item_id))
        if item is None:
            return True, -1, f"item {item_id} does not exist"
        try:
            weight = int(item["weight"])
            value = int(item["value"])
        except (KeyError, TypeError, ValueError):
            return True, -1, f"item {item_id} has invalid weight or value"
        if weight < 0 or value < 0:
            return True, -1, f"item {item_id} has negative weight or value"
        total_weight += weight
        total_value += value

    if total_weight > capacity:
        return True, -1, f"total weight {total_weight} exceeds capacity {capacity}"

    score = maximization_score(total_value, ground_truth)
    return False, score, f"total value: {total_value}, ground truth: {ground_truth}"
