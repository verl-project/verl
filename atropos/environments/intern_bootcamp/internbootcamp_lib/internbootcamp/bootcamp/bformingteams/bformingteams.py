"""# 

### 谜题描述
One day n students come to the stadium. They want to play football, and for that they need to split into teams, the teams must have an equal number of people.

We know that this group of people has archenemies. Each student has at most two archenemies. Besides, if student A is an archenemy to student B, then student B is an archenemy to student A.

The students want to split so as no two archenemies were in one team. If splitting in the required manner is impossible, some students will have to sit on the bench.

Determine the minimum number of students you will have to send to the bench in order to form the two teams in the described manner and begin the game at last.

Input

The first line contains two integers n and m (2 ≤ n ≤ 100, 1 ≤ m ≤ 100) — the number of students and the number of pairs of archenemies correspondingly.

Next m lines describe enmity between students. Each enmity is described as two numbers ai and bi (1 ≤ ai, bi ≤ n, ai ≠ bi) — the indexes of the students who are enemies to each other. Each enmity occurs in the list exactly once. It is guaranteed that each student has no more than two archenemies.

You can consider the students indexed in some manner with distinct integers from 1 to n.

Output

Print a single integer — the minimum number of students you will have to send to the bench in order to start the game.

Examples

Input

5 4
1 2
2 4
5 3
1 4


Output

1

Input

6 2
1 4
3 4


Output

0

Input

6 6
1 2
2 3
3 1
4 5
5 6
6 4


Output

2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = [int(i) for i in raw_input().split()]

pairs = [set() for i in xrange(n)]
for i in xrange(m):
    a, b = [int(i)-1 for i in raw_input().split()]
    pairs[a].add(b)
    pairs[b].add(a)

rest = set(range(n))
bench = 0
mod = 0

while len(rest) > 0:
    prev = start = rest.pop()
    curr = pairs[prev].pop() if len(pairs[prev]) else None
    ct = 1
    while curr:
        rest.discard(curr)
        pairs[curr].discard(prev)
        prev = curr
        curr = pairs[prev].pop() if len(pairs[prev]) else None
        ct += 1
        if start == curr:
            bench += ct % 2
            break
    else:
        prev = start
        curr = pairs[prev].pop() if len(pairs[prev]) else None
        while curr:
            rest.discard(curr)
            pairs[curr].discard(prev)
            prev = curr
            curr = pairs[prev].pop() if len(pairs[prev]) else None
            ct += 1
        mod = (mod + ct) % 2
print bench+mod
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Bformingteamsbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=100, m_max=100):
        self.n_min = n_min
        self.n_max = n_max
        self.m_max = m_max
    
    def case_generator(self):
        max_attempts = 1000  # 防止无限循环
        for _ in range(max_attempts):
            try:
                n = random.randint(self.n_min, self.n_max)
                m_max = min(self.m_max, n)
                if m_max < 1:
                    m_max = 1  # 确保至少一个边
                m = random.randint(1, m_max)

                edges = []
                # 生成环或链结构
                if m == n:
                    # 生成一个环
                    cycle = list(range(1, n + 1))
                    for i in range(n):
                        a = cycle[i]
                        b = cycle[(i + 1) % n]
                        edges.append((a, b) if a < b else (b, a))
                    edges = sorted(edges)
                else:
                    # 生成一个链，需要确保m+1 <=n
                    if m + 1 > n:
                        continue  # 当前n无法生成m条边，跳过
                    chain_nodes = list(range(1, m + 2))  # m+1个节点
                    for i in range(m):
                        edges.append((chain_nodes[i], chain_nodes[i + 1]))

                # 验证边数和度数约束
                if len(edges) != m:
                    continue
                degree = {}
                valid = True
                for a, b in edges:
                    degree[a] = degree.get(a, 0) + 1
                    degree[b] = degree.get(b, 0) + 1
                    if degree[a] > 2 or degree[b] > 2:
                        valid = False
                        break
                if not valid:
                    continue
                # 确保边无重复
                unique_edges = set(tuple(sorted(e)) for e in edges)
                if len(unique_edges) != m:
                    continue

                # 计算正确答案
                pairs = [set() for _ in range(n)]
                for a, b in edges:
                    ai, bi = a - 1, b - 1
                    pairs[ai].add(bi)
                    pairs[bi].add(ai)
                rest = set(range(n))
                bench = 0
                mod = 0
                while rest:
                    start = rest.pop()
                    prev = start
                    ct = 1
                    curr = pairs[prev].pop() if pairs[prev] else None
                    processed = {start}
                    while curr is not None and curr not in processed:
                        processed.add(curr)
                        rest.discard(curr)
                        pairs[curr].discard(prev)
                        next_curr = pairs[curr].pop() if pairs[curr] else None
                        prev, curr = curr, next_curr
                        ct += 1
                    if curr == start and ct > 1:  # 环处理
                        bench += ct % 2
                    else:  # 链处理
                        mod += ct
                mod %= 2
                correct = bench + mod
                return {
                    'n': n,
                    'm': m,
                    'edges': edges,
                    'correct_output': correct
                }
            except Exception as e:
                continue
        # 多次尝试后仍无法生成则返回默认值
        return {
            'n': 5,
            'm': 4,
            'edges': [(1,2),(2,4),(5,3),(1,4)],
            'correct_output': 1
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        edges = '\n'.join(f"{a} {b}" for a, b in question_case['edges'])
        return f"""You are organizing a football game with {n} students. Students have mutual archenemies (each has ≤2). Split into two equal teams with no enemies in the same team. If impossible, some must bench. Find the minimal number to bench.

Input:
{n} {m}
{edges}

Output: A single integer. Place your answer within [answer] tags, e.g., [answer]1[/answer]."""

    @staticmethod
    def extract_output(output):
        import re
        answers = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answers:
            return None
        try:
            return int(answers[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_output']
