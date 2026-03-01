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
n,d,s,q,su=input(),[],[],[],0
for i in range(0,n):
    a,b=map(int,raw_input().split(' '))
    d.append(a)
    s.append(b)
    su+=a
    if(a==1):
        q.append(i)
print su/2
lenth=len(q)
ll=0
while ll<lenth:
    t=q[ll]
    if(d[t]==1):
        print t,s[t]
        s[s[t]]^=t
        d[s[t]]-=1
        if(d[s[t]]==1):
            q.append(s[t])
            lenth+=1
    ll+=1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Amishaandforestbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 8)  # 默认顶点数设为8以保证多样性
        self.tree_prob = params.get('tree_prob', 0.3)  # 控制树结构密度
    
    def case_generator(self):
        n = self.n
        parent = list(range(n))
        edges = []
        nodes = list(range(n))
        random.shuffle(nodes)

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        # 阶段1: 随机生成基础森林结构
        for i in range(n):
            # 每个顶点独立决定是否连接（提升森林多样性）
            if random.random() < self.tree_prob:
                candidates = [j for j in range(n) if j != i]
                random.shuffle(candidates)
                for j in candidates:
                    if find(i) != find(j):
                        edges.append((i, j))
                        parent[find(j)] = find(i)
                        break

        # 阶段2: 动态添加有效边（保证不形成环）
        max_attempts = min(n*2, 1000)  # 平衡效率与生成能力
        for _ in range(max_attempts):
            u, v = random.sample(range(n), 2)
            if find(u) != find(v):
                edges.append((u, v))
                parent[find(v)] = find(u)

        # 计算邻接关系和题目参数
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        data = []
        for v in range(n):
            degree = len(adj[v])
            s = 0
            for neighbor in adj[v]:
                s ^= neighbor
            data.append((degree, s))

        return {
            "n": n,
            "data": data,
            "edges": [tuple(sorted(e)) for e in edges]
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["n"]
        data = question_case["data"]
        prompt = f"""作为图论专家，请根据顶点的度数（degree）和相邻顶点异或和（s）还原原始森林。

**规则说明**
1. 森林由多个无环树组成，没有重复边
2. 顶点编号范围：0到{n-1}
3. 输出边时顶点顺序无关，且边的排列顺序不限

**输入数据**
{n}
""" + "\n".join(f"{d} {s}" for d, s in data) + """

**输出要求**
第一行：边数m
后续m行：每行两个顶点编号（空格分隔）

请将最终答案放在[answer]和[/answer]之间，例如：
[answer]
2
0 1
2 3
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL|re.IGNORECASE)
        if not answer_blocks:
            return None
        
        last_answer = answer_blocks[-1].strip()
        lines = [l.strip() for l in last_answer.split('\n') if l.strip()]
        
        if len(lines) < 1:
            return None
        
        try:
            m = int(lines[0])
            if len(lines) != m + 1:
                return None
            
            edges = []
            for line in lines[1:]:
                parts = list(map(int, line.split()))
                if len(parts) != 2 or parts[0] == parts[1]:
                    return None
                edges.append((parts[0], parts[1]))
            return edges
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 验证边数量匹配
            if len(solution) != len(identity["edges"]):
                return False
            
            # 转换边为规范格式
            solution_set = {tuple(sorted(e)) for e in solution}
            expected_set = set(identity["edges"])
            
            # 检查边集合一致性
            return solution_set == expected_set
        except:
            return False
