"""# 

### 谜题描述
You are given a tree consisting of n vertices. A tree is an undirected connected acyclic graph.

<image> Example of a tree.

You have to paint each vertex into one of three colors. For each vertex, you know the cost of painting it in every color.

You have to paint the vertices so that any path consisting of exactly three distinct vertices does not contain any vertices with equal colors. In other words, let's consider all triples (x, y, z) such that x ≠ y, y ≠ z, x ≠ z, x is connected by an edge with y, and y is connected by an edge with z. The colours of x, y and z should be pairwise distinct. Let's call a painting which meets this condition good.

You have to calculate the minimum cost of a good painting and find one of the optimal paintings. If there is no good painting, report about it.

Input

The first line contains one integer n (3 ≤ n ≤ 100 000) — the number of vertices.

The second line contains a sequence of integers c_{1, 1}, c_{1, 2}, ..., c_{1, n} (1 ≤ c_{1, i} ≤ 10^{9}), where c_{1, i} is the cost of painting the i-th vertex into the first color.

The third line contains a sequence of integers c_{2, 1}, c_{2, 2}, ..., c_{2, n} (1 ≤ c_{2, i} ≤ 10^{9}), where c_{2, i} is the cost of painting the i-th vertex into the second color.

The fourth line contains a sequence of integers c_{3, 1}, c_{3, 2}, ..., c_{3, n} (1 ≤ c_{3, i} ≤ 10^{9}), where c_{3, i} is the cost of painting the i-th vertex into the third color.

Then (n - 1) lines follow, each containing two integers u_j and v_j (1 ≤ u_j, v_j ≤ n, u_j ≠ v_j) — the numbers of vertices connected by the j-th undirected edge. It is guaranteed that these edges denote a tree.

Output

If there is no good painting, print -1.

Otherwise, print the minimum cost of a good painting in the first line. In the second line print n integers b_1, b_2, ..., b_n (1 ≤ b_i ≤ 3), where the i-th integer should denote the color of the i-th vertex. If there are multiple good paintings with minimum cost, print any of them.

Examples

Input


3
3 2 3
4 3 2
3 1 3
1 2
2 3


Output


6
1 3 2 


Input


5
3 4 2 1 2
4 2 1 5 4
5 3 2 1 1
1 2
3 2
4 3
5 3


Output


-1


Input


5
3 4 2 1 2
4 2 1 5 4
5 3 2 1 1
1 2
3 2
4 3
5 4


Output


9
1 3 2 1 3 

Note

All vertices should be painted in different colors in the first example. The optimal way to do it is to paint the first vertex into color 1, the second vertex — into color 3, and the third vertex — into color 2. The cost of this painting is 3 + 2 + 1 = 6.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from collections import defaultdict
from itertools import permutations


def fast2():
    import os, sys, atexit
    from cStringIO import StringIO as BytesIO
    # range = xrange
    sys.stdout = BytesIO()
    atexit.register(lambda: os.write(1, sys.stdout.getvalue()))
    return BytesIO(os.read(0, os.fstat(0).st_size)).readline


class graph:
    # initialize graph
    def __init__(self, gdict=None):
        if gdict is None:
            gdict = defaultdict(list)
        self.gdict, self.edges, self.l = gdict, defaultdict(int), [0] * (n + 1)

    # add edge
    def addEdge(self, node1, node2, w=None):
        self.gdict[node1].append(node2)
        self.gdict[node2].append(node1)
        self.l[node1] += 1
        self.l[node2] += 1

    def dfsUtil(self, v, per):
        global ans, out
        stack, self.visit = [[v, 0]], [0] * (n + 1)
        self.visit[v], tem = per[0] + 1, color[per[0]][v - 1]

        while (stack):
            s, c = stack.pop()

            for i in self.gdict[s]:
                if not self.visit[i]:
                    val = (c + 1) % 3
                    stack.append([i, val])
                    self.visit[i] = per[val] + 1
                    tem += color[per[val]][i - 1]

        if tem < ans:
            ans, out = tem, self.visit[1:]

    def dfs(self):
        v = 0

        for i in range(1, n + 1):
            if self.l[i] == 1:
                v = i
                break

        for per in permutations([0, 1, 2], 3):
            self.dfsUtil(v, per)


input = fast2()
rints = lambda: [int(x) for x in input().split()]
n = int(input())
color = [rints() for _ in range(3)]
g, ans, out = graph(), float('inf'), []

for _ in range(n - 1):
    u, v = rints()
    g.addEdge(u, v)

if list(filter(lambda x: x > 2, g.l)):
    print(-1)
else:
    g.dfs()
    print('%d\n%s' % (ans, ' '.join(map(str, out))))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from itertools import permutations
from collections import defaultdict
from bootcamp import Basebootcamp

class Dpaintthetreebootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=10, chain_prob=0.5, cost_min=1, cost_max=10):
        self.params = {
            'min_n': min_n,
            'max_n': max_n,
            'chain_prob': chain_prob,
            'cost_min': cost_min,
            'cost_max': cost_max
        }
    
    def case_generator(self):
        params = self.params
        n = random.randint(params['min_n'], params['max_n'])
        
        # 生成树结构
        if random.random() < params['chain_prob'] or n < 4:
            # 生成随机链式结构
            nodes = list(range(1, n+1))
            random.shuffle(nodes)
            edges = []
            for i in range(len(nodes)-1):
                edges.append((nodes[i], nodes[i+1]))
        else:
            # 生成带有度数>2节点的树（必然无解）
            edges = []
            if n >= 4:
                edges.extend([(1,2), (1,3), (1,4)])
                current = 4
                for i in range(5, n+1):
                    edges.append((current, i))
                    current = i
        
        # 生成颜色成本
        c1 = [random.randint(params['cost_min'], params['cost_max']) for _ in range(n)]
        c2 = [random.randint(params['cost_min'], params['cost_max']) for _ in range(n)]
        c3 = [random.randint(params['cost_min'], params['cost_max']) for _ in range(n)]
        
        # 计算期望解
        expected_cost, expected_colors = self._solve_puzzle(n, c1, c2, c3, edges)
        
        return {
            'n': n,
            'c1': c1,
            'c2': c2,
            'c3': c3,
            'edges': edges,
            'expected_cost': expected_cost,
            'expected_colors': expected_colors
        }
    
    @staticmethod
    def prompt_func(question_case):
        edges_list = '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        return f"""给定一个包含 {question_case['n']} 个顶点的树结构。需要将每个顶点染色为1/2/3，满足任何三个连续顶点的颜色不同。

颜色成本：
颜色1：{question_case['c1']}
颜色2：{question_case['c2']}
颜色3：{question_case['c3']}

边列表：
{edges_list}

请计算最小成本并给出染色方案，格式示例：
[answer]
<总成本>
<颜色序列>
[/answer]
若无解返回-1。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_block = matches[-1].strip()
        lines = [line.strip() for line in answer_block.split('\n') if line.strip()]
        
        if not lines or lines[0] == '-1':
            return {'cost': -1, 'colors': None}
        
        try:
            cost = int(lines[0])
            colors = []
            if len(lines) >= 2:
                colors = list(map(int, lines[1].split()))
                if any(c not in {1,2,3} for c in colors):
                    return None
            return {'cost': cost, 'colors': colors}
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        # 预期无解的情况验证
        if identity['expected_cost'] == -1:
            return solution.get('cost') == -1 and solution.get('colors') is None
        
        # 成本验证
        if solution['cost'] != identity['expected_cost']:
            return False
        
        # 颜色序列验证
        colors = solution['colors']
        if len(colors) != identity['n'] or any(c not in {1,2,3} for c in colors):
            return False
        
        # 全局路径检查
        adj = defaultdict(list)
        for u, v in identity['edges']:
            adj[u].append(v)
            adj[v].append(u)
        
        # 寻找路径端点
        start = None
        for node in adj:
            if len(adj[node]) == 1:
                start = node
                break
        
        # 遍历路径生成顺序
        path = []
        visited = set()
        stack = [(start, None)]
        while stack:
            node, parent = stack.pop()
            visited.add(node)
            path.append(node)
            neighbors = [n for n in adj[node] if n != parent]
            if neighbors:
                stack.append((neighbors[0], node))
        
        # 检查连续三元组
        for i in range(len(path)-2):
            a, b, c = colors[path[i]-1], colors[path[i+1]-1], colors[path[i+2]-1]
            if len({a, b, c}) < 3:
                return False
        
        return True

    @staticmethod
    def _solve_puzzle(n, c1, c2, c3, edges):
        # 验证树结构合法性
        adj = defaultdict(list)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degrees[u] += 1
            degrees[v] += 1
        
        if any(d > 2 for d in degrees.values()):
            return (-1, None)
        
        # 寻找路径端点
        start = next((node for node in adj if len(adj[node]) == 1), None)
        if not start:
            return (-1, None)
        
        # 动态规划求解
        min_cost = float('inf')
        best_pattern = []
        
        for pattern in permutations([0, 1, 2]):
            current = start
            prev = None
            total = 0
            color_seq = [0]*(n+1)
            color_idx = 0
            
            while True:
                color = pattern[color_idx%3]
                total += [c1[current-1], c2[current-1], c3[current-1]][color]
                color_seq[current] = color + 1
                
                # 移动到下一个节点
                next_nodes = [n for n in adj[current] if n != prev]
                if not next_nodes:
                    break
                prev = current
                current = next_nodes[0]
                color_idx += 1
                
            if total < min_cost:
                min_cost = total
                best_pattern = color_seq[1:]  # 去除0索引
        
        return (min_cost, best_pattern) if min_cost != float('inf') else (-1, None)
