"""# 

### 谜题描述
During the loading of the game \"Dungeons and Candies\" you are required to get descriptions of k levels from the server. Each description is a map of an n × m checkered rectangular field. Some cells of the field contain candies (each cell has at most one candy). An empty cell is denoted as \".\" on the map, but if a cell has a candy, it is denoted as a letter of the English alphabet. A level may contain identical candies, in this case the letters in the corresponding cells of the map will be the same.

<image>

When you transmit information via a network, you want to minimize traffic — the total size of the transferred data. The levels can be transmitted in any order. There are two ways to transmit the current level A:

  1. You can transmit the whole level A. Then you need to transmit n·m bytes via the network. 
  2. You can transmit the difference between level A and some previously transmitted level B (if it exists); this operation requires to transmit dA, B·w bytes, where dA, B is the number of cells of the field that are different for A and B, and w is a constant. Note, that you should compare only the corresponding cells of levels A and B to calculate dA, B. You cannot transform the maps of levels, i.e. rotate or shift them relatively to each other. 



Your task is to find a way to transfer all the k levels and minimize the traffic.

Input

The first line contains four integers n, m, k, w (1 ≤ n, m ≤ 10; 1 ≤ k, w ≤ 1000). Then follows the description of k levels. Each level is described by n lines, each line contains m characters. Each character is either a letter of the English alphabet or a dot (\".\"). Please note that the case of the letters matters.

Output

In the first line print the required minimum number of transferred bytes.

Then print k pairs of integers x1, y1, x2, y2, ..., xk, yk, describing the way to transfer levels. Pair xi, yi means that level xi needs to be transferred by way yi. If yi equals 0, that means that the level must be transferred using the first way, otherwise yi must be equal to the number of a previously transferred level. It means that you will transfer the difference between levels yi and xi to transfer level xi. Print the pairs in the order of transferring levels. The levels are numbered 1 through k in the order they follow in the input.

If there are multiple optimal solutions, you can print any of them.

Examples

Input

2 3 3 2
A.A
...
A.a
..C
X.Y
...


Output

14
1 0
2 1
3 1


Input

1 1 4 1
A
.
B
.


Output

3
1 0
2 0
4 2
3 0


Input

1 3 5 2
ABA
BBB
BBA
BAB
ABB


Output

11
1 0
3 1
2 3
4 2
5 1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import random
#dsu
p = []
def finds(v):
    if p[v] != v:
        p[v] = finds(p[v])
    return p[v]
def union(v1,v2):
    r1 = finds(v1)
    r2 = finds(v2)
    if r1 != r2:
        if random.choice([0,1]) == 0:
            p[r1] = r2
        else:
            p[r2] = r1
#input
I = lambda:map(int,raw_input().split())
n,m,k,w = I()
#solution
t = [[None for _ in xrange(n)] for _ in xrange(k)]
for i in xrange(k):
    for j in xrange(n):
            t[i][j] = list(raw_input())
def dif(a,b):
    ans = 0
    for i in xrange(n):
        for j in xrange(m):
            ans += int(a[i][j] != b[i][j])
    return ans * w
#G = [[] for _ in range(k)]
edge = []
for i in xrange(k):
    for j in xrange(i + 1,k):
        wig = dif (t[i],t[j])
        if wig < n*m:
            edge.append((i,j,wig))
            #G[i].append((j,wig))
            #G[j].append((i,wig))
edge.sort(key=lambda x: x[2]) # sort by the wig parameter
#Kruskal's algorithm
p = list(xrange(0,k)) # parent array
ans = 0
g = [[] for _ in xrange(k)]
for i in edge:
    a,b,w = i
    if finds(a) != finds(b):
        g[a].append(b)
        g[b].append(a)
        union(a,b)
        ans += w
mark = [-1] * k
def dfs(v):
    mark[v] = 1
    for u in g[v]:
        if mark[u] == -1:
            print u + 1,v + 1
            dfs(u)
ans += n * m * len(set([finds(x) for x in xrange(k)]))
print (ans)
for i in xrange(k):
    if mark[i] == -1:
        print i + 1,0
        dfs(i)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cdungeonsandcandiesbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', random.randint(1, 10))
        self.m = params.get('m', random.randint(1, 10))
        self.k = params.get('k', random.randint(1, 5))
        self.w = params.get('w', random.randint(1, 1000))
    
    def case_generator(self):
        levels = []
        for _ in range(self.k):
            level = [''.join(random.choices(
                ['.', 'A', 'B', 'C', 'a', 'b', 'c', 'X', 'Y', 'Z'], k=self.m
            )) for _ in range(self.n)]
            levels.append(level)
        
        correct_total = self._compute_min_traffic(self.n, self.m, self.k, self.w, levels)
        return {
            'n': self.n,
            'm': self.m,
            'k': self.k,
            'w': self.w,
            'levels': levels,
            'correct_total': correct_total
        }
    
    @staticmethod
    def _compute_min_traffic(n, m, k, w, levels):
        def dif(a, b):
            return sum(c1 != c2 for row_a, row_b in zip(a, b) for c1, c2 in zip(row_a, row_b))
        
        edges = []
        for i in range(k):
            for j in range(i+1, k):
                cost = dif(levels[i], levels[j]) * w
                if cost < n * m:
                    edges.append((i, j, cost))
        edges.sort(key=lambda x: x[2])
        
        parent = list(range(k))
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        def union(u, v):
            u_root, v_root = find(u), find(v)
            if u_root != v_root:
                if random.choice([True, False]):
                    parent[u_root] = v_root
                else:
                    parent[v_root] = u_root
        
        mst_cost = 0
        for u, v, cost in edges:
            if find(u) != find(v):
                mst_cost += cost
                union(u, v)
        
        roots = {find(x) for x in range(k)}
        return mst_cost + len(roots) * n * m
    
    @staticmethod
    def prompt_func(question_case) -> str:
        levels = '\n\n'.join([f"Level {i+1}:\n" + '\n'.join(level) 
                            for i, level in enumerate(question_case['levels'])])
        return f"""You are optimizing data transfer for the game "Dungeons and Candies". Transmit {question_case['k']} {question_case['n']}x{question_case['m']} grids with minimal traffic. Each cell contains '.' or a letter (case-sensitive).

Rules:
1. Transmit full level ({question_case['n']*question_case['m']} bytes) or difference from a previous level (d*{question_case['w']} bytes, d=differing cells).
2. Levels can be transmitted in any order.

Output format:
- First line: Total bytes
- Next {question_case['k']} lines: "xi yi" (xi=level number, yi=0 or previous level)

Examples:
[answer]
14
1 0
2 1
3 1
[/answer]

Provide your answer within [answer] tags. Current levels:
{levels}"""
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        lines = solution.split('\n')
        if len(lines) != identity['k'] + 1:
            return False
        
        try:
            claimed = int(lines[0])
            pairs = [tuple(map(int, line.split())) for line in lines[1:]]
        except:
            return False
        
        if {x for x, _ in pairs} != set(range(1, identity['k']+1)):
            return False
        
        transmitted = set()
        total = 0
        for x, y in pairs:
            if y != 0 and y not in transmitted:
                return False
            if y == 0:
                total += identity['n'] * identity['m']
            else:
                diff = sum(c1 != c2 for r1, r2 in zip(identity['levels'][x-1], identity['levels'][y-1]) 
                          for c1, c2 in zip(r1, r2))
                total += diff * identity['w']
            transmitted.add(x)
        
        return total == claimed == identity['correct_total']
