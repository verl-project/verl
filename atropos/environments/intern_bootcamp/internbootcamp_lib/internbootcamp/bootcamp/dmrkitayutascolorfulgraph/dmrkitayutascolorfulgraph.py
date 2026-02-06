"""# 

### 谜题描述
Mr. Kitayuta has just bought an undirected graph with n vertices and m edges. The vertices of the graph are numbered from 1 to n. Each edge, namely edge i, has a color ci, connecting vertex ai and bi.

Mr. Kitayuta wants you to process the following q queries.

In the i-th query, he gives you two integers - ui and vi.

Find the number of the colors that satisfy the following condition: the edges of that color connect vertex ui and vertex vi directly or indirectly.

Input

The first line of the input contains space-separated two integers - n and m(2 ≤ n ≤ 105, 1 ≤ m ≤ 105), denoting the number of the vertices and the number of the edges, respectively.

The next m lines contain space-separated three integers - ai, bi(1 ≤ ai < bi ≤ n) and ci(1 ≤ ci ≤ m). Note that there can be multiple edges between two vertices. However, there are no multiple edges of the same color between two vertices, that is, if i ≠ j, (ai, bi, ci) ≠ (aj, bj, cj).

The next line contains a integer- q(1 ≤ q ≤ 105), denoting the number of the queries.

Then follows q lines, containing space-separated two integers - ui and vi(1 ≤ ui, vi ≤ n). It is guaranteed that ui ≠ vi.

Output

For each query, print the answer in a separate line.

Examples

Input

4 5
1 2 1
1 2 2
2 3 1
2 3 3
2 4 3
3
1 2
3 4
1 4


Output

2
1
0


Input

5 7
1 5 1
2 5 1
3 5 1
4 5 1
1 2 2
2 3 2
3 4 2
5
1 5
5 1
2 5
1 5
1 4


Output

1
1
1
1
2

Note

Let's consider the first sample. 

<image> The figure above shows the first sample. 

  * Vertex 1 and vertex 2 are connected by color 1 and 2. 
  * Vertex 3 and vertex 4 are connected by color 3. 
  * Vertex 1 and vertex 4 are not connected by any single color. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys


def main():
    n = 0
    m = 0
    queries = []

    for i, line in enumerate(sys.stdin):
        if i < 1:
            n, m = map(int, line.split())
            d = [[0] * n for x in range(n)]
            for row in d:
                for index, col in enumerate(row):
                    row[index] = set()
            continue
        if i <= m:
            u, v, c = map(int, line.split())
            d[u-1][v-1].add(c)
            d[v-1][u-1].add(c)
            continue
        if i == m + 1:
            continue
        u, v = map(int, line.split())
        queries.append((u, v, ))

    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = d[i][j].union(d[i][k].intersection(d[k][j]))

    for u, v in queries:
        print len(d[u-1][v-1])


if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from typing import Dict, List, Any
import re
from bootcamp import Basebootcamp

class Dmrkitayutascolorfulgraphbootcamp(Basebootcamp):
    def __init__(self, n: int = 5, m: int = 7, q: int = 5, c_max: int = 3):
        self.n = n
        self.m = m
        self.q = q
        self.c_max = c_max
    
    def case_generator(self) -> Dict:
        n = self.n
        m = self.m
        c_max = self.c_max
        edges = []
        color_edges = {}
        
        for _ in range(m):
            a = random.randint(1, n)
            b = random.randint(1, n)
            while a == b:
                b = random.randint(1, n)
            if a > b:
                a, b = b, a
            c = random.randint(1, c_max)
            if c not in color_edges:
                color_edges[c] = []
            if (a, b) not in color_edges[c]:
                edges.append((a, b, c))
                color_edges[c].append((a, b))
        
        queries = []
        for _ in range(self.q):
            u = random.randint(1, n)
            v = random.randint(1, n)
            while u == v:
                v = random.randint(1, n)
            queries.append((u, v))
        
        case = {
            'n': n,
            'm': m,
            'edges': edges,
            'queries': queries,
            'color_edges': color_edges
        }
        return case

    @staticmethod
    def prompt_func(question_case: Dict) -> str:
        n = question_case['n']
        m = question_case['m']
        edges = question_case['edges']
        queries = question_case['queries']
        
        edges_str = "\n".join(f"{a}-{b} (颜色 {c})" for a, b, c in edges)
        queries_str = "\n".join(f"{u} {v}" for u, v in queries)
        
        prompt = f"Mr. Kitayuta有一个无向图，包含{n}个顶点和{m}条边。边如下：\n{edges_str}\n现在有{len(queries)}个查询，分别是：\n{queries_str}\n对于每个查询(u, v)，请输出满足条件的颜色数目，每个答案用空格分隔，并放在[answer]标签中。"
        return prompt

    @staticmethod
    def extract_output(output: str) -> List[int]:
        match = re.search(r'\[answer\]([\d\s]+)\[\/answer\]', output, re.DOTALL)
        if not match:
            return None
        solutions_str = match.group(1).strip()
        if not solutions_str:
            return None
        try:
            solutions = list(map(int, solutions_str.split()))
        except ValueError:
            return None
        return solutions

    @classmethod
    def _verify_correction(cls, solution: List[int], identity: Dict) -> bool:
        if len(solution) != len(identity['queries']):
            return False
        
        color_edges = identity['color_edges']
        n = identity['n']
        correct = True
        
        for i, (u, v) in enumerate(identity['queries']):
            count = 0
            for c in color_edges:
                parent = list(range(n + 1))
                
                def find(u):
                    while parent[u] != u:
                        parent[u] = parent[parent[u]]
                        u = parent[u]
                    return u
                
                for a, b in color_edges[c]:
                    root_a = find(a)
                    root_b = find(b)
                    if root_a != root_b:
                        parent[root_b] = root_a
                
                if find(u) == find(v):
                    count += 1
            if solution[i] != count:
                correct = False
                break
        
        return correct
