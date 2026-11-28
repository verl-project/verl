"""# 

### 谜题描述
Petya loves lucky numbers. We all know that lucky numbers are the positive integers whose decimal representations contain only the lucky digits 4 and 7. For example, numbers 47, 744, 4 are lucky and 5, 17, 467 are not.

One day Petya encountered a tree with n vertexes. Besides, the tree was weighted, i. e. each edge of the tree has weight (a positive integer). An edge is lucky if its weight is a lucky number. Note that a tree with n vertexes is an undirected connected graph that has exactly n - 1 edges.

Petya wondered how many vertex triples (i, j, k) exists that on the way from i to j, as well as on the way from i to k there must be at least one lucky edge (all three vertexes are pairwise distinct). The order of numbers in the triple matters, that is, the triple (1, 2, 3) is not equal to the triple (2, 1, 3) and is not equal to the triple (1, 3, 2). 

Find how many such triples of vertexes exist.

Input

The first line contains the single integer n (1 ≤ n ≤ 105) — the number of tree vertexes. Next n - 1 lines contain three integers each: ui vi wi (1 ≤ ui, vi ≤ n, 1 ≤ wi ≤ 109) — the pair of vertexes connected by the edge and the edge's weight.

Output

On the single line print the single number — the answer.

Please do not use the %lld specificator to read or write 64-bit numbers in С++. It is recommended to use the cin, cout streams or the %I64d specificator.

Examples

Input

4
1 2 4
3 1 2
1 4 7


Output

16


Input

4
1 2 4
1 3 47
1 4 7447


Output

24

Note

The 16 triples of vertexes from the first sample are: (1, 2, 4), (1, 4, 2), (2, 1, 3), (2, 1, 4), (2, 3, 1), (2, 3, 4), (2, 4, 1), (2, 4, 3), (3, 2, 4), (3, 4, 2), (4, 1, 2), (4, 1, 3), (4, 2, 1), (4, 2, 3), (4, 3, 1), (4, 3, 2).

In the second sample all the triples should be counted: 4·3·2 = 24.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict,Counter

n=input()
d=defaultdict(list)
for i in range(n-1):
    u,v,w=map(int,raw_input().split())
    d1=Counter(str(w))
    if d1['4']+d1['7']!=len(str(w)):
        d[u].append(v)
        d[v].append(u)

vis = [0]*(n+1)
ans=0
for node in range(1,n+1):
    if not vis[node]:
        c=1
        q=[node]
        vis[node]=1
        while q:
            x=q.pop()
            for i in d[x]:
                if not vis[i]:
                    q.append(i)
                    vis[i]=1
                    c+=1
        tot = n-c
        #print tot,c,node
        ans+=((tot-1)*tot*c)
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Eluckytreebootcamp(Basebootcamp):
    def __init__(self, max_n=10**5):
        self.max_n = max_n  # 允许自定义最大节点数

    def case_generator(self):
        """生成具有灵活结构的测试案例"""
        # 扩展n的取值范围，包含边界情形
        n = random.randint(1, min(100, self.max_n))  # 覆盖n=1,2等边界情况
        
        # 生成连通块结构
        k = random.randint(1, max(1, n//2)) if n > 1 else 1  # 确保合理分块
        cs = [1] * k
        for _ in range(n - k):
            cs[random.randint(0, k-1)] += 1
        
        # 生成树结构
        edges = self._generate_tree_structure(n, cs)
        return {'n': n, 'edges': edges}

    def _generate_tree_structure(self, n, cs):
        """生成包含分块结构的树"""
        blocks = []
        current = 1
        for size in cs:
            blocks.append(list(range(current, current + size)))
            current += size

        edges = []
        # 生成块内非幸运边（随机树结构）
        for block in blocks:
            if len(block) > 1:
                self._connect_components(block, edges, self._generate_non_lucky_number)

        # 生成块间幸运边（随机树结构）
        if len(blocks) > 1:
            representatives = [blk[0] for blk in blocks]
            self._connect_components(representatives, edges, self._generate_lucky_number)
        
        return edges

    def _connect_components(self, nodes, edges, weight_generator):
        """通用化的连通组件连接方法"""
        random.shuffle(nodes)
        connected = {nodes[0]}
        for node in nodes[1:]:
            parent = random.choice(list(connected))
            edges.append({
                'u': parent,
                'v': node,
                'w': weight_generator()
            })
            connected.add(node)

    def _generate_lucky_number(self):
        return int(''.join(random.choice('47') for _ in range(random.randint(1, 4))))

    def _generate_non_lucky_number(self):
        while True:
            num = random.randint(1, 10**9)
            if any(d not in {'4','7'} for d in str(num)):
                return num

    @staticmethod
    def prompt_func(case):
        """增强问题描述生成"""
        problem = [
            f"给定一棵包含 {case['n']} 个节点的树，边信息如下：",
            *[f"边 {i+1}: {e['u']}-{e['v']} 权重 {e['w']}" for i, e in enumerate(case['edges'])],
            "计算满足以下条件的有序三元组 (i,j,k) 的总数：",
            "1. i, j, k 互不相同",
            "2. 从i到j的路径包含至少一条幸运边（由4/7组成的权重）",
            "3. 从i到k的路径也包含至少一条幸运边",
            "答案格式：[answer]你的答案[/answer]"
        ]
        return '\n'.join(problem)

    @staticmethod
    def extract_output(text):
        """加强答案提取鲁棒性"""
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', text)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """精确验证算法"""
        # 构建非幸运边图
        graph = defaultdict(list)
        for edge in identity['edges']:
            if not all(c in {'4','7'} for c in str(edge['w'])):
                u, v = edge['u'], edge['v']
                graph[u].append(v)
                graph[v].append(u)

        visited = set()
        total = 0
        n = identity['n']
        
        for node in range(1, n+1):
            if node not in visited:
                # 计算连通分量大小
                stack = [node]
                visited.add(node)
                component_size = 1
                while stack:
                    current = stack.pop()
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                            component_size += 1
                # 累加当前分量的贡献值
                others = n - component_size
                total += component_size * others * (others - 1)
        
        try:
            return int(solution) == total
        except:
            return False
