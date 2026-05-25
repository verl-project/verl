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

import ast
import re
from collections import defaultdict
from collections.abc import Iterable


def answer_payload(answer: str) -> str:
    if "Answer:" not in answer:
        raise ValueError("invalid answer: no 'Answer:' in answer")
    return answer.rsplit("Answer:", 1)[-1].strip()


def parse_answer_literal(answer: str):
    payload = answer_payload(answer)
    try:
        return ast.literal_eval(payload)
    except (ValueError, SyntaxError):
        first_bracket = payload.find("[")
        last_bracket = payload.rfind("]")
        if first_bracket == -1 or last_bracket == -1 or first_bracket >= last_bracket:
            raise ValueError("invalid literal format") from None
        return ast.literal_eval(payload[first_bracket : last_bracket + 1])


def parse_int_list(answer: str, *, allow_empty: bool = False) -> list[int]:
    payload = answer_payload(answer)

    try:
        parsed = parse_answer_literal(answer)
        if isinstance(parsed, list | tuple | set):
            values = list(parsed)
        else:
            raise ValueError("answer must be a list")
    except (ValueError, SyntaxError):
        payload = payload.splitlines()[0].strip().replace("'", "").replace('"', "")
        if "->" in payload:
            payload = ",".join(part.strip() for part in payload.split("->"))
        else:
            match = re.search(r"[{\[\(]([^)}\]]*)[}\])]", payload)
            if match:
                payload = match.group(1)
            else:
                payload = payload.strip("[](){}")
        values = [part.strip() for part in payload.split(",") if part.strip()]

    if not values and not allow_empty:
        raise ValueError("answer list cannot be empty")

    ints = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid integer IDs")
        ints.append(int(value))
    return ints


def normalize_adjacency(graph: dict) -> dict[int, set[int]]:
    neighbors = defaultdict(set)
    vertices = set()

    for raw_u, adjacent in graph.items():
        try:
            u = int(raw_u)
        except (TypeError, ValueError):
            continue

        vertices.add(u)
        if isinstance(adjacent, dict):
            iterator = adjacent.items()
        elif isinstance(adjacent, Iterable) and not isinstance(adjacent, str | bytes):
            iterator = ((raw_v, 1) for raw_v in adjacent)
        else:
            continue

        for raw_v, weight in iterator:
            try:
                v = int(raw_v)
            except (TypeError, ValueError):
                continue
            vertices.add(v)
            if isinstance(weight, int | float) and weight == 0:
                continue
            neighbors[u].add(v)
            neighbors[v].add(u)

    for vertex in vertices:
        neighbors[vertex] = set(neighbors[vertex])
    return dict(neighbors)


def maximization_score(actual: float, optimal: float | int | None) -> float:
    if optimal is None:
        return 0.0
    optimal = float(optimal)
    if optimal == 0:
        return 1.0 if actual == 0 else 0.0
    return actual / optimal


def minimization_score(optimal: float | int | None, actual: float) -> float:
    if optimal is None:
        return 0.0
    optimal = float(optimal)
    if actual == 0:
        return 1.0 if optimal == 0 else 0.0
    return optimal / actual
