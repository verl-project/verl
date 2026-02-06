"""# 

### 谜题描述
Ivan is a novice painter. He has n dyes of different colors. He also knows exactly m pairs of colors which harmonize with each other.

Ivan also enjoy playing chess. He has 5000 rooks. He wants to take k rooks, paint each of them in one of n colors and then place this k rooks on a chessboard of size 10^{9} × 10^{9}.

Let's call the set of rooks on the board connected if from any rook we can get to any other rook in this set moving only through cells with rooks from this set. Assume that rooks can jump over other rooks, in other words a rook can go to any cell which shares vertical and to any cell which shares horizontal.

Ivan wants his arrangement of rooks to have following properties:

  * For any color there is a rook of this color on a board;
  * For any color the set of rooks of this color is connected;
  * For any two different colors a b union of set of rooks of color a and set of rooks of color b is connected if and only if this two colors harmonize with each other.



Please help Ivan find such an arrangement.

Input

The first line of input contains 2 integers n, m (1 ≤ n ≤ 100, 0 ≤ m ≤ min(1000,    (n(n-1))/(2))) — number of colors and number of pairs of colors which harmonize with each other.

In next m lines pairs of colors which harmonize with each other are listed. Colors are numbered from 1 to n. It is guaranteed that no pair occurs twice in this list.

Output

Print n blocks, i-th of them describes rooks of i-th color.

In the first line of block print one number a_{i} (1 ≤ a_{i} ≤ 5000) — number of rooks of color i. In each of next a_{i} lines print two integers x and y (1 ≤ x,    y ≤ 10^{9}) — coordinates of the next rook.

All rooks must be on different cells.

Total number of rooks must not exceed 5000.

It is guaranteed that the solution exists.

Examples

Input

3 2
1 2
2 3


Output

2
3 4
1 4
4
1 2
2 2
2 4
5 4
1
5 1


Input

3 3
1 2
2 3
3 1


Output

1
1 1
1
1 2
1
1 3


Input

3 1
1 3


Output

1
1 1
1
2 2
1
3 1

Note

Rooks arrangements for all three examples (red is color 1, green is color 2 and blue is color 3).

<image>

<image>

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split())
d = {}
for i in range(1, n + 1):
    d[i] = [(i, i)]
for i in range(m):
    x, y = map(int, raw_input().split())
    d[y].append((n + i + 1, y))
    d[x].append((n + i + 1, x))
for i in d:
    print len(d[i])
    for j in d[i]:
        print j[0], j[1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from typing import List, Tuple, Set, Dict, Any
import re

class Ccoloredrooksbootcamp(Basebootcamp):
    def __init__(self, n_min: int = 3, n_max: int = 5, m_ratio: float = 0.5):
        self.n_min = max(n_min, 1)
        self.n_max = n_max
        self.m_ratio = m_ratio

    def case_generator(self) -> dict:
        n = random.randint(self.n_min, self.n_max)
        m = 0
        edges = []
        if n == 1:
            pass  # m remains 0
        else:
            max_possible_m = n * (n - 1) // 2
            m_candidate = min(max_possible_m, 1000)
            m = random.randint(0, m_candidate)
            possible_pairs = [(i+1, j+1) for i in range(n) for j in range(i+1, n)]
            if possible_pairs:
                m = min(m, len(possible_pairs))
                edges = random.sample(possible_pairs, m)
        
        return {
            'n': n,
            'm': m,
            'edges': edges
        }

    @staticmethod
    def prompt_func(question_case: dict) -> str:
        n = question_case['n']
        m = question_case['m']
        edges = question_case['edges']
        input_lines = [f"{n} {m}"] + [f"{x} {y}" for x, y in edges]
        input_example = '\n'.join(input_lines)
        problem = f"""Ivan is a novice painter with {n} different dyes and {m} pairs of harmonizing colors. He needs to place rooks on a chessboard under these conditions:

1. Each color must have at least one rook.
2. All rooks of the same color must form a connected set.
3. Rooks of two different colors are connected if and only if they harmonize.

Input:
{input_example}

Output n blocks. Each block starts with the number of rooks for that color, followed by their coordinates. All coordinates must be unique and total rooks ≤ 5000. Place your answer between [answer] and [/answer] tags."""
        return problem

    @staticmethod
    def extract_output(output: str) -> List[List[Tuple[int, int]]] | None:
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_content = matches[-1].strip()
        lines = [line.strip() for line in answer_content.split('\n') if line.strip()]
        blocks = []
        current_line = 0
        while current_line < len(lines):
            if current_line >= len(lines):
                return None
            try:
                a_i = int(lines[current_line])
            except:
                return None
            if a_i < 1 or a_i > 5000:
                return None
            current_line += 1
            coords = []
            for _ in range(a_i):
                if current_line >= len(lines):
                    return None
                parts = lines[current_line].split()
                if len(parts) != 2:
                    return None
                try:
                    x = int(parts[0])
                    y = int(parts[1])
                    if x < 1 or x > 10**9 or y < 1 or y > 10**9:
                        return None
                    coords.append((x, y))
                except:
                    return None
                current_line += 1
            blocks.append(coords)
        return blocks

    @classmethod
    def _verify_correction(cls, solution: List[List[Tuple[int, int]]], identity: dict) -> bool:
        if not isinstance(solution, list):
            return False
        n = identity['n']
        edges = identity['edges']
        if len(solution) != n:
            return False
        for coords in solution:
            if len(coords) < 1:
                return False
        all_coords = []
        for coords in solution:
            all_coords.extend(coords)
        if len(all_coords) > 5000:
            return False
        if len(all_coords) != len(set(all_coords)):
            return False

        for coords in solution:
            if not cls._is_connected(coords):
                return False

        harmonious_pairs = {frozenset((a, b)) for a, b in edges}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = i + 1, j + 1
                merged = solution[i] + solution[j]
                merged_connected = cls._is_connected(merged)
                expected = frozenset((a, b)) in harmonious_pairs
                if merged_connected != expected:
                    return False
        return True

    @staticmethod
    def _is_connected(coords: List[Tuple[int, int]]) -> bool:
        if not coords:
            return False
        parent = {}

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u

        for x, y in coords:
            r = ('R', x)
            c = ('C', y)
            if r not in parent:
                parent[r] = r
            if c not in parent:
                parent[c] = c
            union(r, c)

        root = find(('R', coords[0][0]))
        for x, y in coords:
            r = ('R', x)
            c = ('C', y)
            if find(r) != root or find(c) != root:
                return False
        return True
