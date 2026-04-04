"""# 

### 谜题描述
Ilya is very fond of graphs, especially trees. During his last trip to the forest Ilya found a very interesting tree rooted at vertex 1. There is an integer number written on each vertex of the tree; the number written on vertex i is equal to ai.

Ilya believes that the beauty of the vertex x is the greatest common divisor of all numbers written on the vertices on the path from the root to x, including this vertex itself. In addition, Ilya can change the number in one arbitrary vertex to 0 or leave all vertices unchanged. Now for each vertex Ilya wants to know the maximum possible beauty it can have.

For each vertex the answer must be considered independently.

The beauty of the root equals to number written on it.

Input

First line contains one integer number n — the number of vertices in tree (1 ≤ n ≤ 2·105).

Next line contains n integer numbers ai (1 ≤ i ≤ n, 1 ≤ ai ≤ 2·105).

Each of next n - 1 lines contains two integer numbers x and y (1 ≤ x, y ≤ n, x ≠ y), which means that there is an edge (x, y) in the tree.

Output

Output n numbers separated by spaces, where i-th number equals to maximum possible beauty of vertex i.

Examples

Input

2
6 2
1 2


Output

6 6 


Input

3
6 2 3
1 2
1 3


Output

6 6 6 


Input

1
10


Output

10 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from fractions import gcd

n = int(stdin.readline().strip())
v = map(int, stdin.readline().strip().split())

adj = [[] for _ in xrange(n)]
for _ in xrange(n-1):
    x, y = map(int, stdin.readline().strip().split())
    adj[x-1].append(y-1)
    adj[y-1].append(x-1)

root_divisors = []
cnt = [0]*200001
d = 1
while d*d <= v[0]:
    if v[0] % d == 0:
        root_divisors.append(d)
        cnt[d] += 1
        if v[0]/d != d:
            root_divisors.append(v[0]/d)
            cnt[v[0]/d] += 1
    d += 1    
s = [0]
visited = [False]*n
visited[0] = True
level = [1]*n
res1 = [0]*n
res2 = [0]*n
res1[0] = v[0]
d = 1
while s:
    x = s[-1]
    any_more = False
    while adj[x]:
        y = adj[x].pop()
        if not visited[y]:
            visited[y] = True
            any_more = True
            s.append(y)
            level[y] = level[x]+1
            res2[y] = gcd(res2[x], v[y])
            for d in root_divisors:
                if v[y] % d == 0:
                    cnt[d] += 1
                if cnt[d] == level[y] or cnt[d] == level[y]-1:
                    res1[y] = max(res1[y], res2[y], d)
            break
    if not any_more:
        s.pop()
        for d in root_divisors:
            if v[x] % d == 0:
                cnt[d] -= 1
print ' '.join(map(str, res1))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
import re
from typing import List, Dict, Any
from collections import defaultdict

class Cilyaandthetreebootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_a=100, **kwargs):
        super().__init__(**kwargs)
        self.max_n = max_n
        self.max_a = max_a

    def case_generator(self) -> Dict[str, Any]:
        """生成有效树结构并计算正确解"""
        n = random.randint(1, self.max_n)
        a = [random.randint(1, self.max_a) for _ in range(n)]
        edges = self._generate_valid_tree(n)
        correct_output = self._compute_solution(n, a, edges)
        
        return {
            'n': n,
            'a': a,
            'edges': edges,
            'correct_output': correct_output
        }

    def _generate_valid_tree(self, n: int) -> List[List[int]]:
        """生成以1为根的合法树结构"""
        if n == 1:
            return []
        
        nodes = list(range(2, n+1))
        random.shuffle(nodes)
        edges = []
        connected = {1}
        for node in nodes:
            parent = random.choice(list(connected))
            edges.append([parent, node])
            connected.add(node)
        return edges

    def _compute_solution(self, n: int, a: List[int], edges: List[List[int]]) -> List[int]:
        """正确实现参考算法逻辑"""
        # 构建邻接表（1-based）
        adj = defaultdict(list)
        for x, y in edges:
            adj[x].append(y)
            adj[y].append(x)

        # 初始化数据结构
        res = [0] * (n+1)  # 1-based索引
        res[1] = a[0]
        cnt = defaultdict(int)
        max_depth = defaultdict(int)
        
        # 预计算根节点所有因数
        root_val = a[0]
        divisors = set()
        d = 1
        while d*d <= root_val:
            if root_val % d == 0:
                divisors.add(d)
                if d != root_val//d:
                    divisors.add(root_val//d)
            d += 1
        
        # 初始化因数计数
        for d in divisors:
            cnt[d] = 1

        # DFS遍历
        stack = [(1, 0, root_val)]  # (current, parent, current_gcd)
        path = []
        
        while stack:
            node, parent, current_gcd = stack.pop()
            path.append(node)
            
            # 计算当前路径长度
            current_depth = len(path)
            
            # 计算当前节点的可能最大值
            max_val = current_gcd
            for d in sorted(divisors, reverse=True):
                if cnt[d] >= current_depth - 1:
                    max_val = max(max_val, d)
                    break
            
            res[node] = max_val
            
            # 处理子节点
            for child in adj[node]:
                if child == parent:
                    continue
                
                # 计算子节点的GCD
                child_gcd = math.gcd(current_gcd, a[child-1])
                
                # 更新因数计数
                for d in divisors:
                    if a[child-1] % d == 0:
                        cnt[d] += 1
                
                stack.append((child, node, child_gcd))
            
            # 回溯时恢复计数
            if path:
                last_node = path.pop()
                for d in divisors:
                    if a[last_node-1] % d == 0:
                        cnt[d] = max(cnt[d]-1, 0)
        
        return [res[i] for i in range(1, n+1)]

    @staticmethod
    def prompt_func(case) -> str:
        problem = (
            "给定根在顶点1的树，每个顶点有整数a_i。定义顶点x的美丽值为根到x路径上的所有数的GCD。"
            "允许将任意一个节点的值改为0或保持不变。对每个顶点求可能的最大美丽值。\n\n"
            f"输入格式：\n第一行：n={case['n']}\n第二行：{' '.join(map(str, case['a']))}\n"
            "接下来的n-1行每行两个整数描述边：\n" + 
            '\n'.join(f"{x} {y}" for x, y in case['edges']) + 
            "\n\n请输出n个用空格分隔的整数，将答案放在[answer]和[/answer]之间。"
        )
        return problem

    @staticmethod
    def extract_output(output: str) -> List[int]:
        matches = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return list(map(int, matches[-1].strip().split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_output']
