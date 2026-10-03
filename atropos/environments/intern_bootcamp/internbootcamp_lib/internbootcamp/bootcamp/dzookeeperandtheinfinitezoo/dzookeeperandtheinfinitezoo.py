"""# 

### 谜题描述
There is a new attraction in Singapore Zoo: The Infinite Zoo.

The Infinite Zoo can be represented by a graph with an infinite number of vertices labeled 1,2,3,…. There is a directed edge from vertex u to vertex u+v if and only if u\&v=v, where \& denotes the [bitwise AND operation](https://en.wikipedia.org/wiki/Bitwise_operation#AND). There are no other edges in the graph.

Zookeeper has q queries. In the i-th query she will ask you if she can travel from vertex u_i to vertex v_i by going through directed edges.

Input

The first line contains an integer q (1 ≤ q ≤ 10^5) — the number of queries.

The i-th of the next q lines will contain two integers u_i, v_i (1 ≤ u_i, v_i < 2^{30}) — a query made by Zookeeper.

Output

For the i-th of the q queries, output \"YES\" in a single line if Zookeeper can travel from vertex u_i to vertex v_i. Otherwise, output \"NO\".

You can print your answer in any case. For example, if the answer is \"YES\", then the output \"Yes\" or \"yeS\" will also be considered as correct answer.

Example

Input


5
1 4
3 6
1 6
6 2
5 5


Output


YES
YES
NO
NO
YES

Note

The subgraph on vertices 1,2,3,4,5,6 is shown below.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
for _ in range(input()):
    u,v = map(int,raw_input().split())
    
    if u>v:
        print \"NO\"
        continue
    
    x,y,f = 0,0,1
    
    for i in range(31):
        if u%2:
            x+=1
        if v%2:
            y+=1
        if y>x:
            f = 0
            break
        u/=2
        v/=2
        
    if f:
        print \"YES\"
    else:
        print \"NO\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Dzookeeperandtheinfinitezoobootcamp(Basebootcamp):
    def __init__(self, u_min=1, u_max=(1 << 30)-1, v_min=1, v_max=(1 << 30)-1):
        self.u_min = u_min
        self.u_max = u_max
        self.v_min = v_min
        self.v_max = v_max

    def case_generator(self):
        # 生成YES案例（可达）的概率调整为40%
        if random.random() < 0.4:
            for _ in range(100):  # 总尝试次数限制
                u = random.randint(self.u_min, self.u_max)
                mask_max = min(u, self.v_max - u)
                if mask_max < 0:
                    continue
                for _ in range(100):  # 单u尝试次数
                    mask = random.randint(0, mask_max)
                    v_prime = u & mask
                    v = u + v_prime
                    if self.v_min <= v <= self.v_max:
                        return {'u': u, 'v': v}
            # 若无法生成有效YES案例，退回生成NO案例
            return self._generate_no_case()
        else:
            return self._generate_no_case()
    
    def _generate_no_case(self):
        """ 专门生成NO案例的方法 """
        # 首先生成u>v的情况（40%概率）
        if random.random() < 0.4:
            v = random.randint(self.v_min, self.u_max-1)
            u = random.randint(v+1, self.u_max)
            return {'u': u, 'v': v}
        # 生成u<=v但不可达的情况（最多尝试200次）
        for _ in range(200):
            u = random.randint(self.u_min, self.u_max)
            v = random.randint(u, self.v_max)
            if not self.is_reachable(u, v):
                return {'u': u, 'v': v}
        # 最终保障机制：生成u>v的简单案例
        v = random.randint(self.v_min, self.u_max-1)
        u = random.randint(v+1, self.u_max)
        return {'u': u, 'v': v}

    @staticmethod
    def prompt_func(question_case):
        u = question_case['u']
        v = question_case['v']
        return f"""Determine if path exists from {u} to {v} in Infinite Zoo.
Rules:
1. Edge u→(u+v') exists iff u & v' = v'
2. Path follows edge directions

Answer format: [answer]YES[/answer] or [answer]NO[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.I)
        return matches[-1].strip().upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution not in {'YES', 'NO'}:
            return False
        return solution == ('YES' if cls.is_reachable(**identity) else 'NO')

    @staticmethod
    def is_reachable(u, v):
        if u > v:
            return False
        x = y = 0
        for _ in range(31):
            x += u & 1
            y += v & 1
            if y > x:
                return False
            u >>= 1
            v >>= 1
        return True
