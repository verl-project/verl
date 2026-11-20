"""# 

### 谜题描述
Twilight Sparkle learnt that the evil Nightmare Moon would return during the upcoming Summer Sun Celebration after one thousand years of imprisonment on the moon. She tried to warn her mentor Princess Celestia, but the princess ignored her and sent her to Ponyville to check on the preparations for the celebration.

<image>

Twilight Sparkle wanted to track the path of Nightmare Moon. Unfortunately, she didn't know the exact path. What she knew is the parity of the number of times that each place Nightmare Moon visited. Can you help Twilight Sparkle to restore any path that is consistent with this information?

Ponyville can be represented as an undirected graph (vertices are places, edges are roads between places) without self-loops and multi-edges. The path can start and end at any place (also it can be empty). Each place can be visited multiple times. The path must not visit more than 4n places.

Input

The first line contains two integers n and m (2 ≤ n ≤ 105; 0 ≤ m ≤ 105) — the number of places and the number of roads in Ponyville. Each of the following m lines contains two integers ui, vi (1 ≤ ui, vi ≤ n; ui ≠ vi), these integers describe a road between places ui and vi.

The next line contains n integers: x1, x2, ..., xn (0 ≤ xi ≤ 1) — the parity of the number of times that each place must be visited. If xi = 0, then the i-th place must be visited even number of times, else it must be visited odd number of times.

Output

Output the number of visited places k in the first line (0 ≤ k ≤ 4n). Then output k integers — the numbers of places in the order of path. If xi = 0, then the i-th place must appear in the path even number of times, else i-th place must appear in the path odd number of times. Note, that given road system has no self-loops, therefore any two neighbouring places in the path must be distinct.

If there is no required path, output -1. If there multiple possible paths, you can output any of them.

Examples

Input

3 2
1 2
2 3
1 1 1


Output

3
1 2 3


Input

5 7
1 2
1 3
1 4
1 5
3 4
3 5
4 5
0 1 0 1 0


Output

10
2 1 3 4 5 4 5 4 3 1 

Input

2 0
0 0


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python


def read():
    nNode, nEdge = map(int, raw_input().split())
    
    adj = [[] for i in range(nNode)]
    
    for i in range(nEdge):
        s, t = map(int, raw_input().split())
        s -= 1
        t -= 1
        adj[s].append(t)
        adj[t].append(s)

    parity = map(int, raw_input().split())
    
    return adj, parity


def rec(pre, cur, visited, adj, parity):

    visited[cur] = True
    parity[cur] = not parity[cur]
    
    ret = [cur]
    
    for child in adj[cur]:
        if visited[child]:
            continue
        
        t = rec(cur, child, visited, adj, parity)

        if t:
            ret += t + [cur]
            parity[cur] = not parity[cur]


    if parity[cur]:
        if pre == -1 or len(ret) == 1:
            ret.pop()
            parity[cur] = not parity[cur]
        else:
            ret.append(pre)
            ret.append(cur)
            parity[cur] = not parity[cur]
            parity[pre] = not parity[pre]

    return ret


def work((adj, parity)):
    visited = [False for i in range(len(adj))]
    
    for node in range(len(adj)):
        if visited[node]:
            continue
        
        ans = rec(-1, node, visited, adj, parity)

        if ans:
            break

    for i in range(len(adj)):
        if not visited[i] and parity[i]:
            print -1
            return
    
    print len(ans)
    for i in range(len(ans)):
        if i != len(ans) - 1:
            print ans[i] + 1,
        else:
            print ans[i] + 1


if __name__ == \"__main__\":
    work(read())
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Clittleponyandsummersuncelebrationbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=5, seed=None):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.seed = seed
        random.seed(seed)

    def generate_valid_case(self):
        """生成合法案例并验证解的存在性"""
        while True:
            n = random.randint(self.n_min, self.n_max)
            m = random.randint(0, min(10, n*(n-1)//2))  # 控制边数规模
            
            # 生成随机图结构
            edges = set()
            adj = [[] for _ in range(n)]
            for _ in range(m):
                while True:
                    u = random.randint(1, n)
                    v = random.randint(1, n)
                    if u != v and (u, v) not in edges and (v, u) not in edges:
                        edges.add((u, v))
                        adj[u-1].append(v-1)
                        adj[v-1].append(u-1)
                        break
            
            # 生成随机奇偶要求（允许无解情况）
            x = [random.choice([0, 1]) for _ in range(n)]
            if random.random() < 0.3:  # 30%概率生成非法案例
                # 强制让某个连通分支的奇偶和为奇数
                comp = random.choice(self.find_components(adj))
                x[comp[0]] ^= 1  # 翻转奇偶性

            # 验证解的存在性
            has_solution = self.check_solution_exists(adj, x)
            solution = self.find_solution(adj, x) if has_solution else None
            
            # 构造案例
            case = {
                'n': n,
                'm': len(edges),
                'edges': sorted((u, v) for u, v in edges),
                'x': x,
                'has_solution': has_solution,
                '_solution': solution
            }
            
            # 确保合法案例至少30%无解
            if case['has_solution'] or (random.random() < 0.3):
                return case

    def find_components(self, adj):
        """寻找连通分支"""
        visited = [False]*len(adj)
        components = []
        for i in range(len(adj)):
            if not visited[i]:
                stack = [i]
                component = []
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        component.append(node)
                        for neighbor in adj[node]:
                            if not visited[neighbor]:
                                stack.append(neighbor)
                components.append(component)
        return components

    def check_solution_exists(self, adj, x):
        """验证解的存在性"""
        components = self.find_components(adj)
        for comp in components:
            parity_sum = sum(x[i] for i in comp) % 2
            if parity_sum != 0:
                return False
        return True

    def find_solution(self, adj, x):
        """查找有效解（参考代码实现）"""
        visited = [False]*len(adj)
        path = []
        
        def dfs(u, parent):
            nonlocal path
            visited[u] = True
            modified = False
            path.append(u+1)
            x[u] ^= 1
            
            for v in adj[u]:
                if not visited[v]:
                    if dfs(v, u):
                        path.append(u+1)
                        x[u] ^= 1
                        modified = True
            
            if x[u]:
                if parent is not None:
                    path.append(parent+1)
                    path.append(u+1)
                    x[parent] ^= 1
                    x[u] ^= 1
                    modified = True
                else:
                    path.pop()
                    x[u] ^= 1
            return modified
        
        for i in range(len(adj)):
            if not visited[i]:
                dfs(i, None)
        
        if len(path) > 4 * len(adj):
            return None
        return path

    def case_generator(self):
        case = self.generate_valid_case()
        return {
            'n': case['n'],
            'm': case['m'],
            'edges': case['edges'],
            'x': case['x'],
            '_solution': case['_solution'],
            'has_solution': case['has_solution']
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']}",
            *[f"{u} {v}" for u, v in question_case['edges']],
            ' '.join(map(str, question_case['x']))
        ]
        input_str = '\n'.join(input_lines)
        return f"""请解决路径奇偶校验问题（Ponyville地图）：
输入：
{input_str}

按以下格式输出答案：
[answer]
<你的解答>
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution.strip() == '-1':
            return not identity['has_solution']
        
        try:
            lines = solution.strip().split('\n')
            k = int(lines[0])
            if k == 0:
                return all(x == 0 for x in identity['x']) and identity['has_solution']
            path = list(map(int, lines[1].split()))
        except:
            return False

        # 基础校验
        n = identity['n']
        if any(not 1 <= v <= n for v in path) or len(path) != k or k > 4*n:
            return False

        # 邻接校验
        edge_set = {(u, v) for u, v in identity['edges']} | {(v, u) for u, v in identity['edges']}
        for i in range(len(path)-1):
            if (path[i], path[i+1]) not in edge_set:
                return False

        # 奇偶校验
        counts = [0]*(n+1)
        for v in path:
            counts[v] += 1
        return all(counts[i+1]%2 == identity['x'][i] for i in range(n))
