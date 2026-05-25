from .common import maximization_score, normalize_adjacency, parse_int_list


def validation(graph, answer, ground_truth):
    """Validate a maximum-independent-set candidate."""
    if not isinstance(graph, dict):
        return True, -1, "graph must be a dictionary"

    try:
        independent_set = parse_int_list(answer, allow_empty=True)
    except ValueError as exc:
        return True, -1, str(exc)

    neighbors = normalize_adjacency(graph)
    if neighbors and not independent_set:
        return True, -1, "independent set cannot be empty for a non-empty graph"
    if len(independent_set) != len(set(independent_set)):
        return True, -1, "independent set contains duplicate vertices"

    graph_nodes = set(neighbors.keys())
    missing = sorted(set(independent_set) - graph_nodes)
    if missing:
        return True, -1, f"vertices are not in the graph: {missing}"

    for index, node in enumerate(independent_set):
        for other in independent_set[index + 1 :]:
            if other in neighbors.get(node, set()):
                return True, -1, f"vertices {node} and {other} are adjacent"

    actual_size = len(independent_set)
    score = maximization_score(actual_size, ground_truth)
    return False, score, f"valid independent set of size {actual_size}, ground truth: {ground_truth}"
