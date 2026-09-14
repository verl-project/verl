"""# 

### 谜题描述
Let's define a forest as a non-directed acyclic graph (also without loops and parallel edges). One day Misha played with the forest consisting of n vertices. For each vertex v from 0 to n - 1 he wrote down two integers, degreev and sv, were the first integer is the number of vertices adjacent to vertex v, and the second integer is the XOR sum of the numbers of vertices adjacent to v (if there were no adjacent vertices, he wrote down 0). 

Next day Misha couldn't remember what graph he initially had. Misha has values degreev and sv left, though. Help him find the number of edges and the edges of the initial graph. It is guaranteed that there exists a forest that corresponds to the numbers written by Misha.

Input

The first line contains integer n (1 ≤ n ≤ 216), the number of vertices in the graph.

The i-th of the next lines contains numbers degreei and si (0 ≤ degreei ≤ n - 1, 0 ≤ si < 216), separated by a space.

Output

In the first line print number m, the number of edges of the graph.

Next print m lines, each containing two distinct numbers, a and b (0 ≤ a ≤ n - 1, 0 ≤ b ≤ n - 1), corresponding to edge (a, b).

Edges can be printed in any order; vertices of the edge can also be printed in any order.

Examples

Input

3
2 3
1 0
1 0


Output

2
1 0
2 0


Input

2
1 1
1 0


Output

1
0 1

Note

The XOR sum of numbers is the result of bitwise adding numbers modulo 2. This operation exists in many modern programming languages. For example, in languages C++, Java and Python it is represented as \"^\", and in Pascal — as \"xor\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
f = lambda: raw_input()
n, r = int(f()), []
a = [map(int, f().split()) for i in range(n)]
g = [i for i, v in enumerate(a) if v[0] == 1]
print sum(v[0] for v in a) / 2
while len(g) > 0:
  _g = []
  for c in g:
    if a[c][0] == 1:
      print a[c][1], c
      a[a[c][1]][0] -= 1
      a[a[c][1]][1] ^= c

      if a[a[c][1]][0] == 1:
        _g.append(a[c][1])

  g = [i for i in _g]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cmishaandforestbootcamp(Basebootcamp):
    def __init__(self, **params):
        # 修正点1：扩大默认参数范围以支持更小的n值
        self.min_n = params.get('min_n', 1)
        self.max_n = params.get('max_n', 10)
        self.n = params.get('n', None)
        
    def case_generator(self):
        if self.n is not None:
            n = self.n
        else:
            n = random.randint(self.min_n, self.max_n)
        
        parent = list(range(n))
        edges = []
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(possible_edges)
        
        # 修正点2：保证m的取值范围覆盖所有合法情况
        max_m = n - 1 if n > 0 else 0
        m = random.randint(0, max_m)
        
        for u, v in possible_edges:
            if len(edges) == m:
                break
            # 使用路径压缩优化并查集
            pu, pv = u, v
            while parent[pu] != pu:
                parent[pu] = parent[parent[pu]]
                pu = parent[pu]
            while parent[pv] != pv:
                parent[pv] = parent[parent[pv]]
                pv = parent[pv]
            if pu != pv:
                parent[pu] = pv
                edges.append((u, v))
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        degrees = [len(neighbors) for neighbors in adj]
        s_list = []
        for i in range(n):
            s = 0
            for neighbor in adj[i]:
                s ^= neighbor
            s_list.append(s)
        
        return {
            'n': n,
            'degrees': degrees,
            's_list': s_list
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        # 保持原有实现不变，格式更规范
        n = question_case['n']
        degrees = question_case['degrees']
        s_list = question_case['s_list']
        
        input_lines = [str(n)]
        for i in range(n):
            input_lines.append(f"{degrees[i]} {s_list[i]}")
        input_str = '\n'.join(input_lines)
        
        prompt = f"""You are a programming assistant. Help Misha reconstruct the original forest based on the degree and XOR sum values of each vertex. The forest is a non-directed acyclic graph (no loops or parallel edges) composed of trees. Each vertex's degree is the number of adjacent vertices, and the XOR sum is the result of XORing the indices of all adjacent vertices.

Input Format:
The first line contains an integer n (1 ≤ n ≤ 2^16), the number of vertices. The next n lines each contain two integers: degree_i and s_i (the degree and XOR sum of vertex i, 0-based).

Output Format:
The first line should be the number of edges m. The following m lines each contain two distinct integers a and b, representing an edge between vertices a and b. The order of edges and the order of vertices in each edge do not matter.

Example Input:
3
2 3
1 0
1 0

Example Output:
2
0 1
0 2

Your task is to solve the following input case. Ensure your answer is enclosed within [answer] and [/answer] tags. Here's the input:

{input_str}"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 增强正则表达式容错能力
        import re
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return None
        try:
            m = int(lines[0])
            edges = []
            for line in lines[1:m+1]:
                parts = list(map(int, line.split()))
                if len(parts) != 2:
                    return None
                a, b = parts
                edges.append((a, b))
            return edges
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 增加边界情况校验
        if solution is None or not isinstance(solution, list):
            return False
        
        n = identity['n']
        expected_degrees = identity['degrees']
        expected_s = identity['s_list']
        edges = solution
        
        # 处理空森林特殊情况
        if n == 0:
            return len(edges) == 0
        
        edge_set = set()
        for a, b in edges:
            if a < 0 or a >= n or b < 0 or b >= n:
                return False
            if a == b:
                return False
            u, v = (a, b) if a < b else (b, a)
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
        
        # 验证边数有效性
        if 2 * len(edges) != sum(expected_degrees):
            return False
        
        # 重建邻接表并验证属性
        adj = [[] for _ in range(n)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # 验证度数和异或和
        for i in range(n):
            if len(adj[i]) != expected_degrees[i]:
                return False
            s = 0
            for neighbor in adj[i]:
                s ^= neighbor
            if s != expected_s[i]:
                return False
        
        # 验证森林结构（无环）
        parent = list(range(n))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        for a, b in edges:
            pa, pb = find(a), find(b)
            if pa == pb:
                return False  # 存在环
            parent[pa] = pb
        
        return True
