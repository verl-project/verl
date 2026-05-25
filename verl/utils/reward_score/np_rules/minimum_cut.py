from .common import minimization_score, parse_answer_literal


def validation(graph, answer, ground_truth):
    """Validate a balanced minimum-bisection solution."""
    if not isinstance(graph, dict):
        return True, -1, "graph must be a dictionary"

    try:
        subsets = parse_answer_literal(answer)
    except (ValueError, SyntaxError) as exc:
        return True, -1, str(exc)

    if not isinstance(subsets, list) or len(subsets) != 2:
        return True, -1, "answer must contain exactly two subsets"
    if not all(isinstance(subset, list) for subset in subsets):
        return True, -1, "each subset must be a list of nodes"

    try:
        subset1 = [int(node) for node in subsets[0]]
        subset2 = [int(node) for node in subsets[1]]
    except (TypeError, ValueError):
        return True, -1, "subsets must contain integer nodes"

    if len(subset1) != len(set(subset1)) or len(subset2) != len(set(subset2)):
        return True, -1, "subsets must not contain duplicate nodes"

    set1, set2 = set(subset1), set(subset2)
    if not set1.isdisjoint(set2):
        return True, -1, f"subsets are not disjoint: {sorted(set1 & set2)}"

    all_nodes = {int(node) for node in graph.keys()}
    union = set1 | set2
    if union != all_nodes:
        missing = sorted(all_nodes - union)
        extra = sorted(union - all_nodes)
        return True, -1, f"partition does not match graph nodes; missing={missing}, extra={extra}"

    size_diff = abs(len(set1) - len(set2))
    if size_diff not in (0, 1) or (len(all_nodes) % 2 == 0 and size_diff != 0):
        return True, -1, f"partition is not balanced: sizes {len(set1)} and {len(set2)}"

    cut_weight = 0.0
    for node in set1:
        for other in set2:
            try:
                cut_weight += float(graph[str(node)].get(str(other), 0))
            except KeyError:
                return True, -1, f"node {node} or {other} is not in the graph"

    score = minimization_score(ground_truth, cut_weight)
    return False, score, f"cut weight: {cut_weight:g}, ground truth: {ground_truth}"
