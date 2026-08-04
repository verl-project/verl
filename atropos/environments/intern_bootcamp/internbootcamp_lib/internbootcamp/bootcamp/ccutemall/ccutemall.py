"""# 

### 谜题描述
You're given a tree with n vertices.

Your task is to determine the maximum possible number of edges that can be removed in such a way that all the remaining connected components will have even size.

Input

The first line contains an integer n (1 ≤ n ≤ 10^5) denoting the size of the tree. 

The next n - 1 lines contain two integers u, v (1 ≤ u, v ≤ n) each, describing the vertices connected by the i-th edge.

It's guaranteed that the given edges form a tree.

Output

Output a single integer k — the maximum number of edges that can be removed to leave all connected components with even size, or -1 if it is impossible to remove edges in order to satisfy this property.

Examples

Input

4
2 4
4 1
3 1


Output

1

Input

3
1 2
1 3


Output

-1

Input

10
7 1
8 4
8 10
4 7
6 5
9 3
3 5
2 10
2 5


Output

4

Input

2
1 2


Output

0

Note

In the first example you can remove the edge between vertices 1 and 4. The graph after that will have two connected components with two vertices in each.

In the second example you can't remove edges in such a way that all components have even number of vertices, so the answer is -1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def main():
  n = input()
  if n % 2 != 0:
    print(-1)
    return
  links = [[1, set()] for i in range(1, n+1)]
  W = 0
  L = 1
  i = 0
  while i < n-1:
    i += 1
    [a, b] = [int(x) for x in raw_input().split()]
    links[a-1][L].add(b-1)
    links[b-1][L].add(a-1)
  count = 0
  sear = 0
  cur = 0
  while sear < n:
    li = cur
    l = links[li]
    if len(l[L]) != 1:
      if sear == cur:
        sear += 1
      cur = sear
      continue
    mi = l[L].pop()
    m = links[mi]
    if l[W] % 2 == 0:
      count += 1
    else:
      m[W] += 1
    m[L].remove(li)
    if mi < sear:
      cur = mi
    else:
      sear += 1
      cur = sear
  print(count)
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ccutemallbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=20, **kwargs):
        self.n_min = max(n_min, 2)
        self.n_max = n_max

    def generate_tree(self, n):
        """使用改进的Prüfer序列生成更平衡的树结构"""
        if n == 1:
            return []
        if n == 2:
            return [(1, 2)]
        
        # 生成更平衡的Prüfer序列
        prufer = []
        for _ in range(n-2):
            # 偏好选择中间节点
            prufer.append(random.randint(max(1, n//4), min(n, 3*n//4)))
        
        degree = [1]*(n+1)
        for node in prufer:
            degree[node] += 1

        edges = []
        for node in prufer:
            for v in range(1, n+1):
                if degree[v] == 1:
                    edges.append((node, v))
                    degree[node] -= 1
                    degree[v] -= 1
                    break

        # 处理剩余节点时保持随机性
        remaining = [v for v in range(1, n+1) if degree[v] == 1]
        edges.append((remaining.pop(), remaining.pop()))
        
        # 随机打乱边并确保节点顺序
        random.shuffle(edges)
        return [(u, v) if u < v else (v, u) for u, v in edges]

    def _calculate_solution(self, n, edges):
        """修正的DFS解法"""
        if n % 2 != 0:
            return -1
        
        # 构建邻接表
        adj = [[] for _ in range(n+1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        self.count = 0
        
        def dfs(node, parent):
            size = 1
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                child_size = dfs(neighbor, node)
                size += child_size
                if child_size % 2 == 0:
                    self.count += 1
            return size
        
        total_size = dfs(1, -1)
        # 验证总大小
        return self.count if total_size % 2 == 0 else -1

    def case_generator(self):
        """优化的案例生成逻辑"""
        while True:
            n = random.randint(self.n_min, self.n_max)
            edges = self.generate_tree(n)
            
            try:
                correct_k = self._calculate_solution(n, edges)
            except:
                continue  # 处理可能的递归深度问题
            
            # 根据奇偶性验证解的有效性
            if (n % 2 == 1 and correct_k == -1) or (n % 2 == 0 and correct_k >= 0):
                return {'n': n, 'edges': edges, 'correct_k': correct_k}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        edges = question_case['edges']
        edge_list = '\n'.join(f"{u} {v}" for u, v in edges)
        return f"""Given a tree with {n} vertices represented by undirected edges. Find the maximum number of edges that can be removed such that all resulting connected components contain an even number of vertices. If impossible, output -1.

Input Format:
n
u1 v1
...
u(n-1) v(n-1)

Current Tree Structure:
{n}
{edge_list}

Present your answer as a single integer within [answer] tags. Example: [answer]2[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['correct_k']
        except:
            return False
