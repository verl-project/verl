"""# 

### 谜题描述
Momiji has got a rooted tree, consisting of n nodes. The tree nodes are numbered by integers from 1 to n. The root has number 1. Momiji decided to play a game on this tree.

The game consists of several steps. On each step, Momiji chooses one of the remaining tree nodes (let's denote it by v) and removes all the subtree nodes with the root in node v from the tree. Node v gets deleted as well. The game finishes when the tree has no nodes left. In other words, the game finishes after the step that chooses the node number 1.

Each time Momiji chooses a new node uniformly among all the remaining nodes. Your task is to find the expectation of the number of steps in the described game.

Input

The first line contains integer n (1 ≤ n ≤ 105) — the number of nodes in the tree. The next n - 1 lines contain the tree edges. The i-th line contains integers ai, bi (1 ≤ ai, bi ≤ n; ai ≠ bi) — the numbers of the nodes that are connected by the i-th edge.

It is guaranteed that the given graph is a tree.

Output

Print a single real number — the expectation of the number of steps in the described game.

The answer will be considered correct if the absolute or relative error doesn't exceed 10 - 6.

Examples

Input

2
1 2


Output

1.50000000000000000000


Input

3
1 2
1 3


Output

2.00000000000000000000

Note

In the first sample, there are two cases. One is directly remove the root and another is remove the root after one step. Thus the expected steps are: 

1 × (1 / 2) + 2 × (1 / 2) = 1.5

In the second sample, things get more complex. There are two cases that reduce to the first sample, and one case cleaned at once. Thus the expected steps are: 

1 × (1 / 3) + (1 + 1.5) × (2 / 3) = (1 / 3) + (5 / 3) = 2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
e = [[] for _ in range(n)]
for _ in range(n - 1):
	a, b = map(int, raw_input().split())
	e[a - 1].append(b - 1)
	e[b - 1].append(a - 1)
d = [0] * n
d[0] = 1
q = [0]
for u in q:
	for v in e[u]:
		if not d[v]:
			d[v] = d[u] + 1
			q.append(v)
print sum(1. / x for x in d)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
import math
from bootcamp import Basebootcamp

class Egameontreebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10):
        self.min_n = max(min_n, 1)
        self.max_n = max(max_n, self.min_n)

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        
        if n == 1:
            return {'n': 1, 'edges': [], 'expected': 1.0}
        
        # 生成随机树的优化算法（保证连通性）
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        root = nodes[0]
        parent_map = {}
        edges = []
        for i in range(1, len(nodes)):
            parent = random.choice(nodes[:i])
            edges.append((parent, nodes[i]))
            parent_map[nodes[i]] = parent
        
        # 确保生成的是有效树结构
        expected = self._compute_expected(n, edges)
        return {
            'n': n,
            'edges': sorted([(min(a,b), max(a,b)) for a,b in edges]),  # 标准化边格式
            'expected': expected
        }

    def _compute_expected(self, n, edges):
        if n == 1:
            return 1.0
            
        # 构建邻接表并计算深度
        adj = [[] for _ in range(n+1)]  # 使用1-based索引
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        depth = [0]*(n+1)
        depth[1] = 1  # 根节点为1
        q = deque([1])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if depth[v] == 0 and v != 1:
                    depth[v] = depth[u] + 1
                    q.append(v)
        return sum(1.0/d for d in depth[1:n+1])  # 只取有效节点

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        edges = question_case['edges']
        
        # 处理输入格式的换行符问题
        input_lines = [str(n)]
        if n > 1:
            input_lines += [f"{a} {b}" for a, b in edges]
        
        # 避免在f-string中直接使用换行符表达式
        input_str = '\n'.join(input_lines)
        example = ""
        if n == 2 and edges == [(1,2)]:
            example = "\n示例输入输出与第一个官方样例一致"
        
        return f"""Momiji有一个根树，包含{n}个节点，根节点是1。每一步随机选择剩余节点并删除其子树，求期望步骤数。

输入数据：
{input_str}

输出要求：
1. 结果保留至少12位小数
2. 将最终答案包裹在[answer]标签内
3. 示例格式：[answer]1.500000000000[/answer]{example}"""

    @staticmethod
    def extract_output(output):
        import re
        pattern = r'\[answer\]\s*([\d.]+(?:e-?\d+)?)\s*\[/answer\]'
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1].replace(' ', ''))
            except ValueError:
                pass
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = identity['expected']
            actual = float(solution)
            return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
        except (ValueError, TypeError):
            return False
