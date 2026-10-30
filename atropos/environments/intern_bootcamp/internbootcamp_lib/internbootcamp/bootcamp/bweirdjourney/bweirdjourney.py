"""# 

### 谜题描述
Little boy Igor wants to become a traveller. At first, he decided to visit all the cities of his motherland — Uzhlyandia.

It is widely known that Uzhlyandia has n cities connected with m bidirectional roads. Also, there are no two roads in the country that connect the same pair of cities, but roads starting and ending in the same city can exist. Igor wants to plan his journey beforehand. Boy thinks a path is good if the path goes over m - 2 roads twice, and over the other 2 exactly once. The good path can start and finish in any city of Uzhlyandia.

Now he wants to know how many different good paths are in Uzhlyandia. Two paths are considered different if the sets of roads the paths goes over exactly once differ. Help Igor — calculate the number of good paths.

Input

The first line contains two integers n, m (1 ≤ n, m ≤ 106) — the number of cities and roads in Uzhlyandia, respectively.

Each of the next m lines contains two integers u and v (1 ≤ u, v ≤ n) that mean that there is road between cities u and v.

It is guaranteed that no road will be given in the input twice. That also means that for every city there is no more than one road that connects the city to itself.

Output

Print out the only integer — the number of good paths in Uzhlyandia.

Examples

Input

5 4
1 2
1 3
1 4
1 5


Output

6

Input

5 3
1 2
2 3
4 5


Output

0

Input

2 2
1 1
1 2


Output

1

Note

In first sample test case the good paths are: 

  * 2 → 1 → 3 → 1 → 4 → 1 → 5, 
  * 2 → 1 → 3 → 1 → 5 → 1 → 4, 
  * 2 → 1 → 4 → 1 → 5 → 1 → 3, 
  * 3 → 1 → 2 → 1 → 4 → 1 → 5, 
  * 3 → 1 → 2 → 1 → 5 → 1 → 4, 
  * 4 → 1 → 2 → 1 → 3 → 1 → 5. 



There are good paths that are same with displayed above, because the sets of roads they pass over once are same: 

  * 2 → 1 → 4 → 1 → 3 → 1 → 5, 
  * 2 → 1 → 5 → 1 → 3 → 1 → 4, 
  * 2 → 1 → 5 → 1 → 4 → 1 → 3, 
  * 3 → 1 → 4 → 1 → 2 → 1 → 5, 
  * 3 → 1 → 5 → 1 → 2 → 1 → 4, 
  * 4 → 1 → 3 → 1 → 2 → 1 → 5, 
  * and all the paths in the other direction. 



Thus, the answer is 6.

In the second test case, Igor simply can not walk by all the roads.

In the third case, Igor walks once over every road.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input
 
inp = [int(x) for x in sys.stdin.read().split()]; ii = 0
 
n = inp[ii]; ii += 1
m = inp[ii]; ii += 1

special = 0
found = [1]*n

coupl = [[] for _ in range(n)]
for _ in range(m):
    u = inp[ii] - 1; ii += 1
    v = inp[ii] - 1; ii += 1
    found[u] = 0
    found[v] = 0

    if u != v:
        coupl[u].append(v)
        coupl[v].append(u)
    else:
        special += 1

root = 0
while found[root]: root += 1
found[root] = 1
bfs = [root]
for node in bfs:
    for nei in coupl[node]:
        if not found[nei]:
            found[nei] = 1
            bfs.append(nei)
 
if all(found):
    print sum(len(c)*(len(c) - 1) for c in coupl)//2 + \
    special * (special - 1)//2 + \
    special * (m - special)
else:
    print 0
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Bweirdjourneybootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
    
    def case_generator(self):
        # 参数处理逻辑优化
        n = self.params.get('n', random.randint(1, 5))
        n = max(1, n)  # 保证最小城市数为1

        # 边数范围修正
        max_possible = n * (n + 1) // 2  # 包含自环的最大可能边数
        min_m = 1 if n == 1 else (n - 1)
        m = self.params.get('m', random.randint(min_m, min(max_possible, 10)))
        m = max(min_m, min(m, max_possible))  # 确保合法范围

        edges = []
        edge_set = set()

        # 处理n=1的特殊情况
        if n == 1:
            # 强制生成自环边
            edges = [(1, 1)] * m
            edge_set = set([(1,1)])
        else:
            # 生成连通图的修正逻辑
            # 生成生成树保证连通性
            nodes = list(range(1, n+1))
            random.shuffle(nodes)
            for i in range(1, n):
                u = nodes[i-1]
                v = nodes[i]
                edge = tuple(sorted((u, v)))
                edges.append((u, v))
                edge_set.add(edge)
            
            # 添加剩余边（可能包括自环）
            remaining = m - len(edges)
            while remaining > 0:
                u = random.randint(1, n)
                v = random.randint(1, n)
                if u == v:
                    if (u, v) not in edge_set:
                        edges.append((u, v))
                        edge_set.add((u, v))
                        remaining -= 1
                else:
                    edge = tuple(sorted((u, v)))
                    if edge not in edge_set:
                        edges.append((u, v))
                        edge_set.add(edge)
                        remaining -= 1

        # 正确计算结果
        expected = self.calculate_good_paths(n, m, edges)
        return {
            "n": n,
            "m": m,
            "edges": edges,
            "expected": expected
        }
    
    @staticmethod
    def prompt_func(question_case):
        edges = question_case['edges']
        edges_str = '\n'.join(f"{u} {v}" for u, v in edges)
        prompt = f"""You are solving a graph theory problem about Bweirdjourney. The country has {question_case['n']} cities connected by {question_case['m']} roads. A valid path must traverse exactly {question_case['m']-2} roads twice and 2 roads once. Paths are considered different if their sets of single-use roads differ.

Road list (u v pairs):
{edges_str}

Calculate the total number of valid good paths. Provide your answer as a single integer within [answer][/answer] tags. Example: [answer]3[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
    
    @staticmethod
    def calculate_good_paths(n, m, edges):
        # 严格遵循参考代码逻辑
        special = 0
        found = [1] * n
        coupl = [[] for _ in range(n)]
        
        for u, v in edges:
            u0 = u - 1
            v0 = v - 1
            found[u0] = 0
            found[v0] = 0
            if u0 != v0:
                coupl[u0].append(v0)
                coupl[v0].append(u0)
            else:
                special += 1
        
        # 连通性检查
        root = 0
        while root < n and found[root]:
            root += 1
        
        if root < n:
            found[root] = 1
            bfs = [root]
            for node in bfs:
                for nei in coupl[node]:
                    if not found[nei]:
                        found[nei] = 1
                        bfs.append(nei)
        
        if not all(found):
            return 0
        
        # 计算结果
        sum_degree = sum(len(c)*(len(c)-1) for c in coupl) // 2
        sum_special = special * (special-1) // 2
        sum_mixed = special * (m - special)
        return sum_degree + sum_special + sum_mixed
