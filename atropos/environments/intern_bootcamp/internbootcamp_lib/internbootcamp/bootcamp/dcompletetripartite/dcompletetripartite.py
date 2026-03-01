"""# 

### 谜题描述
You have a simple undirected graph consisting of n vertices and m edges. The graph doesn't contain self-loops, there is at most one edge between a pair of vertices. The given graph can be disconnected.

Let's make a definition.

Let v_1 and v_2 be two some nonempty subsets of vertices that do not intersect. Let f(v_{1}, v_{2}) be true if and only if all the conditions are satisfied:

  1. There are no edges with both endpoints in vertex set v_1. 
  2. There are no edges with both endpoints in vertex set v_2. 
  3. For every two vertices x and y such that x is in v_1 and y is in v_2, there is an edge between x and y. 



Create three vertex sets (v_{1}, v_{2}, v_{3}) which satisfy the conditions below;

  1. All vertex sets should not be empty. 
  2. Each vertex should be assigned to only one vertex set. 
  3. f(v_{1}, v_{2}), f(v_{2}, v_{3}), f(v_{3}, v_{1}) are all true. 



Is it possible to create such three vertex sets? If it's possible, print matching vertex set for each vertex.

Input

The first line contains two integers n and m (3 ≤ n ≤ 10^{5}, 0 ≤ m ≤ min(3 ⋅ 10^{5}, (n(n-1))/(2))) — the number of vertices and edges in the graph.

The i-th of the next m lines contains two integers a_{i} and b_{i} (1 ≤ a_{i} < b_{i} ≤ n) — it means there is an edge between a_{i} and b_{i}. The graph doesn't contain self-loops, there is at most one edge between a pair of vertices. The given graph can be disconnected.

Output

If the answer exists, print n integers. i-th integer means the vertex set number (from 1 to 3) of i-th vertex. Otherwise, print -1.

If there are multiple answers, print any.

Examples

Input


6 11
1 2
1 3
1 4
1 5
1 6
2 4
2 5
2 6
3 4
3 5
3 6


Output


1 2 2 3 3 3 

Input


4 6
1 2
1 3
1 4
2 3
2 4
3 4


Output


-1

Note

In the first example, if v_{1} = \{ 1 \}, v_{2} = \{ 2, 3 \}, and v_{3} = \{ 4, 5, 6 \} then vertex sets will satisfy all conditions. But you can assign vertices to vertex sets in a different way; Other answers like \"2 3 3 1 1 1\" will be accepted as well.

<image>

In the second example, it's impossible to make such vertex sets.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
from atexit import register
from io import BytesIO
sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))
input = lambda: sys.stdin.readline().rstrip('\r\n')
n,m = map(int,input().split(\" \"))
connected = [[]for i in range(n+1)]
edges = []
for i in range(m):
    a,b = map(int,input().split(\" \"))
    connected[a].append(b)
    connected[b].append(a)
    edges.append((a,b))
flag = True
labels = [0]*(n+1)
labels[1] = 1
alln = set(range(1,n+1))
if len(connected[1]) >0:
    node2 = connected[1][0]
else:
    flag = False
if flag:
    s1 = alln-set(connected[1])
    s2 = alln-set(connected[node2])
    s3 = alln-s1-s2
    n1,n2,n3 = len(s1),len(s2),len(s3)
    if n1+n2+n3!=n or min(n1,n2,n3) == 0:
        flag = False
    for i in list(s1):
        labels[i] = 1
    for i in list(s2):
        labels[i] = 2
    for i in list(s3):
        labels[i] = 3
    cnts = [0]*3
    for a,b in edges:
        if labels[a] == labels[b]:
            flag = False
            break
        cnts[labels[a]-1] += 1
        cnts[labels[b]-1] += 1
    if flag and cnts[0] == n1*(n2+n3) and cnts[1] == n2*(n1+n3) and cnts[2] == n3*(n1+n2):
        out = []
        for i in range(1,n+1):
            out.append(str(labels[i]))
        print \" \".join(out)
    else:
        print -1
else:
    print -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random

class Dcompletetripartitebootcamp(Basebootcamp):
    def __init__(self):
        super().__init__()
    
    def case_generator(self):
        # 生成符合题目要求的n范围 (3 ≤ n ≤ 1e5)
        n = random.randint(3, 20)  # 示例使用较小范围便于测试，实际可调至1e5
        
        # 确保三个子集都至少有一个顶点
        sizes = [1, 1, 1]
        remaining = n - 3
        for _ in range(remaining):
            sizes[random.randint(0, 2)] += 1
        
        # 随机分配顶点到三个子集
        vertices = list(range(1, n+1))
        random.shuffle(vertices)
        
        v1 = sorted(vertices[:sizes[0]])
        v2 = sorted(vertices[sizes[0]:sizes[0]+sizes[1]])
        v3 = sorted(vertices[sizes[0]+sizes[1]:])
        
        # 计算跨子集边的总数（数学公式直接计算）
        m = sizes[0]*sizes[1] + sizes[1]*sizes[2] + sizes[2]*sizes[0]
        
        # 生成边集合（仅记录跨子集的边）
        edges = []
        # V1-V2边
        for a in v1:
            edges.extend((min(a,b), max(a,b)) for b in v2)
        # V2-V3边
        for b in v2:
            edges.extend((min(b,c), max(b,c)) for c in v3)
        # V3-V1边
        for c in v3:
            edges.extend((min(c,a), max(c,a)) for a in v1)
        
        # 去重并排序
        edges = sorted(list(set(edges)))
        
        return {
            "n": n,
            "m": len(edges),
            "edges": edges,
            "expected_sets": {
                1: v1,
                2: v2,
                3: v3
            }
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [f"{question_case['n']} {question_case['m']}"]
        input_lines.extend(f"{a} {b}" for a, b in question_case['edges'])
        input_block = "\n".join(input_lines)
        
        prompt = f"""Given an undirected graph with {question_case['n']} vertices and {question_case['m']} edges, determine if the vertices can be partitioned into three non-empty subsets meeting these conditions:

1. All vertices are in exactly one subset
2. For each subset pair (V1,V2), (V2,V3), (V3,V1):
   - No internal edges within either subset
   - Complete bipartite connections between subsets

Input:
{input_block}

Output format:
If possible: {question_case['n']} space-separated integers (1/2/3)
If impossible: -1

Place your final answer between [answer] and [/answer]. Example:
[answer]1 2 2 3 3 3[/answer]"""

        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 快速拒绝错误格式
        if not solution or solution.strip() == "-1":
            return False
        
        try:
            labels = list(map(int, solution.split()))
        except ValueError:
            return False
        
        # 验证基础条件
        if len(labels) != identity['n']:
            return False
        if not all(l in {1,2,3} for l in labels):
            return False
        
        # 构建分组映射
        groups = {1: set(), 2: set(), 3: set()}
        for v, l in enumerate(labels, 1):
            groups[l].add(v)
        
        # 检查非空子集
        if any(len(groups[i]) == 0 for i in [1,2,3]):
            return False
        
        # 验证边约束
        edge_set = set(map(tuple, identity['edges']))
        
        # 验证每个子集的内部无连接
        for group in groups.values():
            for a in group:
                for b in group:
                    if a < b and (a, b) in edge_set:
                        return False
        
        # 验证跨子集的完全连接
        expected_pairs = [
            (groups[1], groups[2]),
            (groups[2], groups[3]),
            (groups[3], groups[1])
        ]
        for s1, s2 in expected_pairs:
            for a in s1:
                for b in s2:
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return False
        
        return True
