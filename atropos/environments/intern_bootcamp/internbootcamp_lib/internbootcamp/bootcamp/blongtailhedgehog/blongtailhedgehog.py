"""# 

### 谜题描述
This Christmas Santa gave Masha a magic picture and a pencil. The picture consists of n points connected by m segments (they might cross in any way, that doesn't matter). No two segments connect the same pair of points, and no segment connects the point to itself. Masha wants to color some segments in order paint a hedgehog. In Mashas mind every hedgehog consists of a tail and some spines. She wants to paint the tail that satisfies the following conditions: 

  1. Only segments already presented on the picture can be painted; 
  2. The tail should be continuous, i.e. consists of some sequence of points, such that every two neighbouring points are connected by a colored segment; 
  3. The numbers of points from the beginning of the tail to the end should strictly increase. 



Masha defines the length of the tail as the number of points in it. Also, she wants to paint some spines. To do so, Masha will paint all the segments, such that one of their ends is the endpoint of the tail. Masha defines the beauty of a hedgehog as the length of the tail multiplied by the number of spines. Masha wants to color the most beautiful hedgehog. Help her calculate what result she may hope to get.

Note that according to Masha's definition of a hedgehog, one segment may simultaneously serve as a spine and a part of the tail (she is a little girl after all). Take a look at the picture for further clarifications.

Input

First line of the input contains two integers n and m(2 ≤ n ≤ 100 000, 1 ≤ m ≤ 200 000) — the number of points and the number segments on the picture respectively. 

Then follow m lines, each containing two integers ui and vi (1 ≤ ui, vi ≤ n, ui ≠ vi) — the numbers of points connected by corresponding segment. It's guaranteed that no two segments connect the same pair of points.

Output

Print the maximum possible value of the hedgehog's beauty.

Examples

Input

8 6
4 5
3 5
2 5
1 2
2 8
6 7


Output

9


Input

4 6
1 2
1 3
1 4
2 3
2 4
3 4


Output

12

Note

The picture below corresponds to the first sample. Segments that form the hedgehog are painted red. The tail consists of a sequence of points with numbers 1, 2 and 5. The following segments are spines: (2, 5), (3, 5) and (4, 5). Therefore, the beauty of the hedgehog is equal to 3·3 = 9.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m = map(int,raw_input().split())
edge = [ [] for _ in range(n)]

for x in range(m):
    u,v = map(lambda x: int(x)-1, raw_input().split())
    edge[u].append(v)
    edge[v].append(u)

inc = [ 1 for _ in range(n)]
for u in range(n):
    for v in edge[u]:
        if u > v:
            inc[u] = max(inc[u], inc[v]+1)

for u in range(n):
    inc[u] *= len(edge[u])
print max(inc)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import itertools
import random
from bootcamp import Basebootcamp

def _calculate_max_beauty(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
    
    inc = [1] * n
    for u in range(n):
        for v in adj[u]:
            if v < u:
                inc[u] = max(inc[u], inc[v] + 1)
    
    max_beauty = max(inc[u] * len(adj[u]) for u in range(n))
    return max_beauty

class Blongtailhedgehogbootcamp(Basebootcamp):  # 修改类名保持一致性
    def __init__(self, n=None, m=None, seed=None):
        super().__init__()
        self.n = n if n and 2 <= n <= 100000 else None
        self.m = m if m and 1 <= m <= 200000 else None
        self.seed = seed
        random.seed(seed) if seed else None

    def case_generator(self):
        """生成保证存在有效解的测试用例"""
        n = self.n or random.randint(5, 10)
        min_edges = n - 1  # 保证图连通
        
        # 生成链状基础结构保证有解
        base_edges = [(i, i+1) for i in range(1, n)]
        remaining = n*(n-1)//2 - len(base_edges)
        
        m = max(min_edges, self.m or random.randint(min_edges, min(2*n, remaining)))
        possible_edges = list(
            itertools.combinations(range(1, n+1), 2)
        )
        possible_edges = list(set(possible_edges) - set(base_edges))
        
        # 添加随机边到基础链
        extra_edges = random.sample(possible_edges, k=m - len(base_edges))
        edges = base_edges + extra_edges
        
        # 清理可能的重复边
        edges = list({tuple(sorted(e)) for e in edges})
        random.shuffle(edges)
        
        return {
            'n': n,
            'm': len(edges),
            'edges': edges,
            'correct_answer': _calculate_max_beauty(n, edges)
        }

    @staticmethod
    def prompt_func(case):
        input_lines = [f"{case['n']} {case['m']}"] + [f"{u} {v}" for u, v in case['edges']]
        return f"""Christmas puzzle challenge:
Given {case['n']} points and {case['m']} segments, find the maximum beauty of a hedgehog configuration.

**Rules:**
1. Tail must be strictly increasing numeric sequence
2. Beauty = (Tail length) × (Spines count at endpoint)
3. Segments can serve dual roles

Input:
\"""
{" ".join(map(str, [case['n'], case['m']]))}
{chr(10).join(f"{u} {v}" for u, v in case['edges'])}
\"""

Calculate the maximum beauty. Put ONLY the numeric answer between [answer] and [/answer] tags."""

    @staticmethod
    def extract_output(text):
        import re
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', text, re.IGNORECASE)
        return int(answers[-1]) if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
