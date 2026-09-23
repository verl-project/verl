"""# 

### 谜题描述
On Children's Day, the child got a toy from Delayyy as a present. However, the child is so naughty that he can't wait to destroy the toy.

The toy consists of n parts and m ropes. Each rope links two parts, but every pair of parts is linked by at most one rope. To split the toy, the child must remove all its parts. The child can remove a single part at a time, and each remove consume an energy. Let's define an energy value of part i as vi. The child spend vf1 + vf2 + ... + vfk energy for removing part i where f1, f2, ..., fk are the parts that are directly connected to the i-th and haven't been removed.

Help the child to find out, what is the minimum total energy he should spend to remove all n parts.

Input

The first line contains two integers n and m (1 ≤ n ≤ 1000; 0 ≤ m ≤ 2000). The second line contains n integers: v1, v2, ..., vn (0 ≤ vi ≤ 105). Then followed m lines, each line contains two integers xi and yi, representing a rope from part xi to part yi (1 ≤ xi, yi ≤ n; xi ≠ yi).

Consider all the parts are numbered from 1 to n.

Output

Output the minimum total energy the child should spend to remove all n parts of the toy.

Examples

Input

4 3
10 20 30 40
1 4
1 2
2 3


Output

40


Input

4 4
100 100 100 100
1 2
2 3
2 4
3 4


Output

400


Input

7 10
40 10 20 10 20 80 40
1 5
4 7
4 5
5 2
5 7
6 4
1 6
1 3
4 3
1 4


Output

160

Note

One of the optimal sequence of actions in the first sample is:

  * First, remove part 3, cost of the action is 20. 
  * Then, remove part 2, cost of the action is 10. 
  * Next, remove part 4, cost of the action is 10. 
  * At last, remove part 1, cost of the action is 0. 



So the total energy the child paid is 20 + 10 + 10 + 0 = 40, which is the minimum.

In the second sample, the child will spend 400 no matter in what order he will remove the parts.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, e = map(int, raw_input().split())
energies = zip(range(1, n+1), map(int, raw_input().split()))
graph = map(lambda a: set(), range(n+1))

for i in range(e):
    x, y = map(int, raw_input().split())
    graph[x].add(y)
    graph[y].add(x)

energies.sort(key=lambda a: a[1])

total = 0
for energy in energies:
    this = energy[0]
    neighbors = graph[this]
    if neighbors:
        total += energy[1] * len(neighbors)
        for neighbor in neighbors:
            graph[neighbor].remove(this)
        graph[this].clear()

print total
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from bootcamp import Basebootcamp
import random

class Cthechildandtoybootcamp(Basebootcamp):
    def __init__(self, n_min=4, n_max=10, v_min=0, v_max=100, m_max_ratio=1.0):
        # 约束参数范围确保符合题目要求
        self.n_min = max(1, min(n_min, 1000))
        self.n_max = max(1, min(n_max, 1000))
        self.v_min = max(0, v_min)
        self.v_max = max(0, v_max)
        self.m_max_ratio = min(1.0, max(0.0, m_max_ratio))

    def case_generator(self):
        # 保证n的范围符合题目要求（1 ≤ n ≤ 1000）
        n = random.randint(self.n_min, self.n_max)
        max_possible_edges = n * (n - 1) // 2
        
        # 双重约束：m_max不超过2000，且不超过可能边数的比例
        m_max = min(
            int(max_possible_edges * self.m_max_ratio),
            2000  # 题目约束m ≤ 2000
        )
        m = random.randint(0, m_max)

        # 生成唯一无向边（保证i < j避免重复）
        possible_edges = []
        for i in range(1, n+1):
            for j in range(i+1, n+1):
                possible_edges.append((i, j))
        random.shuffle(possible_edges)
        edges = possible_edges[:m]

        v = [random.randint(self.v_min, self.v_max) for _ in range(n)]
        return {
            'n': n,
            'm': m,
            'v': v,
            'edges': edges
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']}",
            ' '.join(map(str, question_case['v']))
        ]
        for x, y in question_case['edges']:
            input_lines.append(f"{x} {y}")
        input_str = '\n'.join(input_lines)
        prompt = f"""你是一个聪明的孩子，想要找到拆除玩具所有部件的最小总能量消耗。玩具由{question_case['n']}个部件和{question_case['m']}条绳子组成。规则如下：
1. 每次只能移除一个未被移除的部件。
2. 移除部件i时，消耗的能量等于其能量值v_i乘以此时仍直接连接的未移除部件数量。
3. 你需要选择移除顺序使得总能量最小。

输入格式：
第一行包含n和m，接下来是n个v值，然后是m行边。

输入数据：
{input_str}

请计算出最小总能量，并将答案放在[answer]和[/answer]之间。例如：[answer]100[/answer]。
"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        try:
            return int(matches[-1]) if matches else None
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 完全按照题目逻辑的验证流程
        n = identity['n']
        v = identity['v']
        edges = identity['edges']

        # 构建邻接表
        graph = {i: set() for i in range(1, n+1)}
        for x, y in edges:
            graph[x].add(y)
            graph[y].add(x)

        # 按v_i从小到大排序
        nodes = sorted([(i+1, val) for i, val in enumerate(v)], key=lambda x: x[1])

        total = 0
        for node_id, vi in nodes:
            # 计算当前连接的未移除部件数量
            active_neighbors = len(graph[node_id])
            total += vi * active_neighbors
            
            # 更新邻接表（双向删除）
            for neighbor in list(graph[node_id]):
                graph[neighbor].remove(node_id)
            graph[node_id].clear()
        
        return solution == total
