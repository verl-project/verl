"""# 

### 谜题描述
Welcome! Everything is fine.

You have arrived in The Medium Place, the place between The Good Place and The Bad Place. You are assigned a task that will either make people happier or torture them for eternity.

You have a list of k pairs of people who have arrived in a new inhabited neighborhood. You need to assign each of the 2k people into one of the 2k houses. Each person will be the resident of exactly one house, and each house will have exactly one resident.

Of course, in the neighborhood, it is possible to visit friends. There are 2k - 1 roads, each of which connects two houses. It takes some time to traverse a road. We will specify the amount of time it takes in the input. The neighborhood is designed in such a way that from anyone's house, there is exactly one sequence of distinct roads you can take to any other house. In other words, the graph with the houses as vertices and the roads as edges is a tree.

The truth is, these k pairs of people are actually soulmates. We index them from 1 to k. We denote by f(i) the amount of time it takes for the i-th pair of soulmates to go to each other's houses.

As we have said before, you will need to assign each of the 2k people into one of the 2k houses. You have two missions, one from the entities in The Good Place and one from the entities of The Bad Place. Here they are:

  * The first mission, from The Good Place, is to assign the people into the houses such that the sum of f(i) over all pairs i is minimized. Let's define this minimized sum as G. This makes sure that soulmates can easily and efficiently visit each other; 
  * The second mission, from The Bad Place, is to assign the people into the houses such that the sum of f(i) over all pairs i is maximized. Let's define this maximized sum as B. This makes sure that soulmates will have a difficult time to visit each other. 



What are the values of G and B?

Input

The first line of input contains a single integer t (1 ≤ t ≤ 500) denoting the number of test cases. The next lines contain descriptions of the test cases.

The first line of each test case contains a single integer k denoting the number of pairs of people (1 ≤ k ≤ 10^5). The next 2k - 1 lines describe the roads; the i-th of them contains three space-separated integers a_i, b_i, t_i which means that the i-th road connects the a_i-th and b_i-th houses with a road that takes t_i units of time to traverse (1 ≤ a_i, b_i ≤ 2k, a_i ≠ b_i, 1 ≤ t_i ≤ 10^6). It is guaranteed that the given roads define a tree structure.

It is guaranteed that the sum of the k in a single file is at most 3 ⋅ 10^5.

Output

For each test case, output a single line containing two space-separated integers G and B. 

Example

Input


2
3
1 2 3
3 2 4
2 4 3
4 5 6
5 6 5
2
1 2 1
1 3 2
1 4 3


Output


15 33
6 6

Note

For the sample test case, we have a minimum sum equal to G = 15. One way this can be achieved is with the following assignment:

  * The first pair of people get assigned to houses 5 and 6, giving us f(1) = 5; 
  * The second pair of people get assigned to houses 1 and 4, giving us f(2) = 6; 
  * The third pair of people get assigned to houses 3 and 2, giving us f(3) = 4. 



Note that the sum of the f(i) is 5 + 6 + 4 = 15. 

We also have a maximum sum equal to B = 33. One way this can be achieved is with the following assignment:

  * The first pair of people get assigned to houses 1 and 4, giving us f(1) = 6; 
  * The second pair of people get assigned to houses 6 and 2, giving us f(2) = 14; 
  * The third pair of people get assigned to houses 3 and 5, giving us f(3) = 13. 



Note that the sum of the f(i) is 6 + 14 + 13 = 33. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import Counter,defaultdict
from sys import stdin, stdout
raw_input = stdin.readline
pr = stdout.write

for t in xrange(input()):
    n=input()
    d=defaultdict(list)
    n*=2
    for i in xrange(n-1):
        u,v,w=map(int, raw_input().split())
        d[u].append((v,w))
        d[v].append((u,w))
    par=[0]*(n+1)
    cst=[0]*(n+1)
    q=[1]
    pos=0
    par[1]=1
    while len(q)>pos:
        x=q[pos]
        pos+=1
        for i,dis in d[x]:
            if par[i]:
                continue
            par[i]=x
            cst[i]=dis
            q.append(i)
    dp=[1]*(n+1)
    mx,mn=0,0
    for i in q[::-1]:
        dp[par[i]]+=dp[i]
        mx+=cst[i]*min(dp[i],n-dp[i])
        mn+=cst[i]*(dp[i]%2)
    pr(str(mn)+' '+str(mx)+'\n')
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict, deque
import random
import re
from bootcamp import Basebootcamp

class Cjeremybearimybootcamp(Basebootcamp):
    def __init__(self, k_min=1, k_max=3, min_weight=1, max_weight=10**6):
        self.k_min = k_min
        self.k_max = k_max
        self.min_weight = min_weight
        self.max_weight = max_weight

    def case_generator(self):
        """使用Prim算法生成合法的随机树，确保全连接"""
        k = random.randint(self.k_min, self.k_max)
        n = 2 * k
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        
        # 使用Prim算法生成树
        visited = set([nodes[0]])
        edges = []
        candidates = []
        
        # 初始化候选边
        current = nodes[0]
        for node in nodes[1:]:
            candidates.append((current, node))
        
        while len(visited) < n:
            # 随机选择一条连接已访问和未访问节点的边
            random.shuffle(candidates)
            found = False
            for i in range(len(candidates)):
                u, v = candidates[i]
                if (u in visited) ^ (v in visited):
                    weight = random.randint(self.min_weight, self.max_weight)
                    edges.append([u, v, weight] if u < v else [v, u, weight])
                    new_node = v if u in visited else u
                    visited.add(new_node)
                    # 添加新候选边
                    for node in nodes:
                        if node not in visited:
                            candidates.append((new_node, node))
                    del candidates[i]
                    found = True
                    break
            if not found and len(visited) < n:
                # 处理不连通情况（理论上不会发生）
                unvisited = list(set(nodes) - visited)
                u = random.choice(list(visited))
                v = random.choice(unvisited)
                weight = random.randint(self.min_weight, self.max_weight)
                edges.append([u, v, weight])
                visited.add(v)

        return {
            'k': k,
            'edges': sorted(edges, key=lambda x: (x[0], x[1]))
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [str(question_case['k'])] + [
            f"{u} {v} {t}" for u, v, t in question_case['edges']
        ]
        input_str = "1\n" + "\n".join(input_lines)

        return f"""Welcome to The Medium Place! Your task is to compute G (minimum possible sum) and B (maximum possible sum) for soulmate pairs.

Problem Description:
- Assign 2k people into 2k houses arranged in a tree structure
- Each road has travel time t_i
- Soulmates must be assigned to two different houses
- Calculate sum of path times between all pairs:
  G: Minimum possible sum
  B: Maximum possible sum

Input:
{input_str}

Format your answer as: [answer]G B[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = re.sub(r'\s+', ' ', matches[-1]).strip()
        parts = last_match.split()
        if len(parts) != 2:
            return None
        try:
            return f"{int(parts[0])} {int(parts[1])}"
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        parts = solution.split()
        if len(parts) != 2:
            return False
        try:
            sol_g, sol_b = map(int, parts)
        except ValueError:
            return False

        correct_g, correct_b = cls.calculate_GB(identity['k'], identity['edges'])
        return sol_g == correct_g and sol_b == correct_b

    @classmethod
    def calculate_GB(cls, k, edges):
        n = 2 * k
        adj = defaultdict(list)
        for u, v, t in edges:
            adj[u].append((v, t))
            adj[v].append((u, t))

        # BFS建立父子关系
        par = [0] * (n + 1)
        cst = [0] * (n + 1)
        q = deque([1])
        par[1] = -1  # 根节点无父

        while q:
            u = q.popleft()
            for v, t in adj[u]:
                if par[v] == 0 and v != par[u]:
                    par[v] = u
                    cst[v] = t
                    q.append(v)

        # 后序遍历计算子树大小
        dp = [1] * (n + 1)
        stack = []
        visited = [False] * (n + 1)
        stack.append((1, False))
        
        while stack:
            node, processed = stack.pop()
            if processed:
                for v, _ in adj[node]:
                    if v != par[node] and par[v] == node:
                        dp[node] += dp[v]
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))
            # 子节点逆序入栈保证处理顺序
            children = [v for v, _ in adj[node] if v != par[node] and par[v] == node]
            for child in reversed(children):
                stack.append((child, False))

        mn = mx = 0
        for v in range(2, n + 1):
            mn += cst[v] * (dp[v] % 2)
            mx += cst[v] * min(dp[v], n - dp[v])

        return mn, mx
