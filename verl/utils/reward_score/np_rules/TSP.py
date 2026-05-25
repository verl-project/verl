from .common import minimization_score, parse_int_list


def validation(graph, answer, ground_truth):
    """Validate a traveling-salesman tour."""
    if not isinstance(graph, dict) or not graph:
        return True, -1, "graph must be a non-empty dictionary"

    try:
        path = parse_int_list(answer)
    except ValueError as exc:
        return True, -1, str(exc)

    if path[0] != path[-1]:
        return True, -1, f"path is not a cycle: start {path[0]} and end {path[-1]} differ"

    cities = {int(city) for city in graph.keys()}
    if len(path) != len(cities) + 1:
        return True, -1, f"path length must be {len(cities) + 1}, got {len(path)}"
    if set(path[:-1]) != cities or len(set(path[:-1])) != len(cities):
        return True, -1, "path must visit each city exactly once before returning"

    total_distance = 0.0
    for index in range(len(path) - 1):
        current, next_city = path[index], path[index + 1]
        try:
            total_distance += float(graph[str(current)][str(next_city)])
        except KeyError:
            return True, -1, f"no distance found between cities {current} and {next_city}"

    score = minimization_score(ground_truth, total_distance)
    return False, score, f"total distance: {total_distance:g}, ground truth: {ground_truth}"
