# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .common import maximization_score, normalize_adjacency, parse_int_list


def validation(graph, answer, ground_truth):
    """Validate a maximum-clique candidate."""
    if not isinstance(graph, dict):
        return True, -1, "graph must be a dictionary"

    try:
        clique = parse_int_list(answer)
    except ValueError as exc:
        return True, -1, str(exc)

    if len(clique) != len(set(clique)):
        return True, -1, "clique contains duplicate vertices"

    neighbors = normalize_adjacency(graph)
    graph_nodes = set(neighbors.keys())
    missing = sorted(set(clique) - graph_nodes)
    if missing:
        return True, -1, f"vertices are not in the graph: {missing}"

    clique_set = set(clique)
    for vertex in clique:
        disconnected = sorted((clique_set - {vertex}) - neighbors.get(vertex, set()))
        if disconnected:
            return True, -1, f"vertex {vertex} is not connected to {disconnected}"

    clique_size = len(clique)
    score = maximization_score(clique_size, ground_truth)
    return False, score, f"valid clique of size {clique_size}, ground truth: {ground_truth}"
