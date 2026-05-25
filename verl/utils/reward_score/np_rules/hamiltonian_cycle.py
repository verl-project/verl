from .common import maximization_score, normalize_adjacency, parse_int_list


def validation(graph, answer, ground_truth):
    """Validate a simple cycle for the Hamiltonian-cycle task."""
    if not isinstance(graph, dict):
        return True, -1, "graph must be a dictionary"

    try:
        path = parse_int_list(answer)
    except ValueError as exc:
        return True, -1, str(exc)

    if path[0] != path[-1]:
        return True, -1, f"path is not a cycle: start {path[0]} and end {path[-1]} differ"

    visited = path[:-1]
    if len(set(visited)) != len(visited):
        return True, -1, "nodes must not repeat except for the closing node"

    neighbors = normalize_adjacency(graph)
    graph_nodes = set(neighbors.keys())
    missing = sorted(set(visited) - graph_nodes)
    if missing:
        return True, -1, f"nodes are not in the graph: {missing}"

    for index in range(len(path) - 1):
        current, next_node = path[index], path[index + 1]
        if next_node not in neighbors.get(current, set()):
            return True, -1, f"nodes {current} and {next_node} are not connected"

    visited_count = len(set(visited))
    score = maximization_score(visited_count, ground_truth)
    return False, score, f"valid cycle visiting {visited_count} nodes, ground truth: {ground_truth}"
