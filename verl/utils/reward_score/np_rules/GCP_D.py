from .common import minimization_score, normalize_adjacency, parse_int_list


def validation(graph, answer, ground_truth):
    """Validate a graph coloring solution."""
    if not isinstance(graph, dict):
        return True, -1, "graph must be a dictionary"

    try:
        colors = parse_int_list(answer)
    except ValueError as exc:
        return True, -1, str(exc)

    neighbors = normalize_adjacency(graph)
    num_vertices = len(neighbors)
    if len(colors) != num_vertices:
        return True, -1, f"invalid coloring: expected {num_vertices} colors, got {len(colors)}"

    if any(color <= 0 for color in colors):
        return True, -1, "colors must be positive integers"

    for node, adjacent in neighbors.items():
        if node < 0 or node >= len(colors):
            return True, -1, f"node {node} is outside the coloring range"
        for neighbor in adjacent:
            if neighbor < 0 or neighbor >= len(colors):
                return True, -1, f"node {neighbor} is outside the coloring range"
            if colors[node] == colors[neighbor]:
                return True, -1, f"nodes {node} and {neighbor} share color {colors[node]}"

    used_colors = len(set(colors))
    score = minimization_score(ground_truth, used_colors)
    return False, score, f"valid coloring with {used_colors} colors, ground truth: {ground_truth}"
