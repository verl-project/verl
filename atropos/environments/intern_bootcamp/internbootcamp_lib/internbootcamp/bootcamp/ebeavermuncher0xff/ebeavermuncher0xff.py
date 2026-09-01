"""# 

### 谜题描述
\"Eat a beaver, save a tree!\" — That will be the motto of ecologists' urgent meeting in Beaverley Hills.

And the whole point is that the population of beavers on the Earth has reached incredible sizes! Each day their number increases in several times and they don't even realize how much their unhealthy obsession with trees harms the nature and the humankind. The amount of oxygen in the atmosphere has dropped to 17 per cent and, as the best minds of the world think, that is not the end.

In the middle of the 50-s of the previous century a group of soviet scientists succeed in foreseeing the situation with beavers and worked out a secret technology to clean territory. The technology bears a mysterious title \"Beavermuncher-0xFF\". Now the fate of the planet lies on the fragile shoulders of a small group of people who has dedicated their lives to science.

The prototype is ready, you now need to urgently carry out its experiments in practice.

You are given a tree, completely occupied by beavers. A tree is a connected undirected graph without cycles. The tree consists of n vertices, the i-th vertex contains ki beavers. 

\"Beavermuncher-0xFF\" works by the following principle: being at some vertex u, it can go to the vertex v, if they are connected by an edge, and eat exactly one beaver located at the vertex v. It is impossible to move to the vertex v if there are no beavers left in v. \"Beavermuncher-0xFF\" cannot just stand at some vertex and eat beavers in it. \"Beavermuncher-0xFF\" must move without stops.

Why does the \"Beavermuncher-0xFF\" works like this? Because the developers have not provided place for the battery in it and eating beavers is necessary for converting their mass into pure energy.

It is guaranteed that the beavers will be shocked by what is happening, which is why they will not be able to move from a vertex of the tree to another one. As for the \"Beavermuncher-0xFF\", it can move along each edge in both directions while conditions described above are fulfilled.

The root of the tree is located at the vertex s. This means that the \"Beavermuncher-0xFF\" begins its mission at the vertex s and it must return there at the end of experiment, because no one is going to take it down from a high place. 

Determine the maximum number of beavers \"Beavermuncher-0xFF\" can eat and return to the starting vertex.

Input

The first line contains integer n — the number of vertices in the tree (1 ≤ n ≤ 105). The second line contains n integers ki (1 ≤ ki ≤ 105) — amounts of beavers on corresponding vertices. Following n - 1 lines describe the tree. Each line contains two integers separated by space. These integers represent two vertices connected by an edge. Vertices are numbered from 1 to n. The last line contains integer s — the number of the starting vertex (1 ≤ s ≤ n).

Output

Print the maximum number of beavers munched by the \"Beavermuncher-0xFF\".

Please, do not use %lld specificator to write 64-bit integers in C++. It is preferred to use cout (also you may use %I64d).

Examples

Input

5
1 3 1 3 2
2 5
3 4
4 5
1 5
4


Output

6


Input

3
2 1 1
3 2
1 2
3


Output

2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()+1
k = [0]+map(int,raw_input().split())
d = [[] for _ in xrange(n)]
for _ in xrange(n-2):
    a,b = [int(x) for x in raw_input().split()]
    d[a].append(b)
    d[b].append(a)
s = input()
q = [0] 
d[0] = [s]
v = [True]+[False]*n
for x in q:
    for u in d[x]:
        if not v[u]:
            v[u]=True
            q.append(u)
r = [[] for _ in xrange(n)]
v = [True]*n
f = [0]*n
for i in xrange(n): 
    if i!=s: k[i]-=1
for x in reversed(q):
    v[x]=False
    for u in d[x]:
        if v[u]: continue
        rx = sorted(r[u],reverse=True)[:k[u]]
        res = sum(rx)+2*len(rx)
        k[u]-=len(rx)
        b = min(k[u],f[u])
        k[u]-=b
        res+=b*2
        f[x]+=k[u]
        r[x].append(res)
print r[0][0]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def calculate_max_beavers(n, k_list, edges, s):
    if n == 1:
        return 0  # 无法移动
    
    # 构建邻接表 (1-based)
    adj = [[] for _ in range(n+1)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    # 确定处理顺序 (BFS分层)
    parent = [0]*(n+1)
    visited = [False]*(n+1)
    q = [s]
    visited[s] = True
    processing_order = [s]
    
    while q:
        u = q.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                q.append(v)
                processing_order.append(v)
    
    # 逆序处理节点
    processing_order.reverse()
    
    # 预处理k值 (除s外所有节点减1)
    k = {i: k_list[i-1] for i in range(1, n+1)}
    for i in range(1, n+1):
        if i != s:
            k[i] -= 1
    
    f = {i: 0 for i in range(1, n+1)}  # 剩余容量
    r = {i: [] for i in range(1, n+1)} # 子树结果
    
    for u in processing_order:
        children = [v for v in adj[u] if parent[u] != v]
        
        for v in children:
            # 处理子树v的结果
            sorted_r = sorted(r[v], reverse=True)
            take = min(len(sorted_r), k[u])
            res = sum(sorted_r[:take]) + 2 * take
            k[u] -= take
            
            # 处理剩余容量
            b = min(k[u], f[v])
            res += 2 * b
            k[u] -= b
            
            f[u] += k[v]  # 累积剩余容量
            r[u].append(res)
    
    return sum(r[s])

class Ebeavermuncher0xffbootcamp(Basebootcamp):
    def __init__(self, max_n=5, min_k=1, max_k=5):
        self.max_n = max_n
        self.min_k = min_k
        self.max_k = max_k
    
    def case_generator(self):
        while True:
            n = random.randint(1, self.max_n)
            k = [random.randint(self.min_k, self.max_k) for _ in range(n)]
            edges = []
            
            # 使用Prim算法生成随机树
            if n > 1:
                nodes = list(range(1, n+1))
                random.shuffle(nodes)
                connected = {nodes[0]}
                while len(connected) < n:
                    u = random.choice(list(connected))
                    v_candidates = [v for v in nodes if v not in connected]
                    if not v_candidates: break
                    v = random.choice(v_candidates)
                    edges.append((u, v))
                    connected.add(v)
            
            s = random.randint(1, n)
            try:
                expected = calculate_max_beavers(n, k, edges, s)
                return {
                    'n': n,
                    'k': k,
                    'edges': edges,
                    's': s,
                    'expected': expected
                }
            except:
                continue
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [
            str(question_case['n']),
            ' '.join(map(str, question_case['k']))
        ]
        for a, b in question_case['edges']:
            input_lines.append(f"{a} {b}")
        input_lines.append(str(question_case['s']))
        input_sample = '\n'.join(input_lines)
        
        return f"""请解决以下Ebeavermuncher0xff问题，将最终答案放在[answer]标签内：

输入：
{input_sample}

规则：
1. 机器必须从顶点s出发并返回
2. 每次移动必须吃目标顶点1个beaver
3. 吃过的顶点必须留有足够beavers供后续移动

答案示例：[answer]6[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        try:
            return int(matches[-1].strip()) if matches else None
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
