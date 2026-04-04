"""# 

### 谜题描述
You are given a graph with n nodes and m directed edges. One lowercase letter is assigned to each node. We define a path's value as the number of the most frequently occurring letter. For example, if letters on a path are \"abaca\", then the value of that path is 3. Your task is find a path whose value is the largest.

Input

The first line contains two positive integers n, m (1 ≤ n, m ≤ 300 000), denoting that the graph has n nodes and m directed edges.

The second line contains a string s with only lowercase English letters. The i-th character is the letter assigned to the i-th node.

Then m lines follow. Each line contains two integers x, y (1 ≤ x, y ≤ n), describing a directed edge from x to y. Note that x can be equal to y and there can be multiple edges between x and y. Also the graph can be not connected.

Output

Output a single line with a single integer denoting the largest value. If the value can be arbitrarily large, output -1 instead.

Examples

Input

5 4
abaca
1 2
1 3
3 4
4 5


Output

3


Input

6 6
xzyabc
1 2
3 1
2 3
5 4
4 3
6 4


Output

-1


Input

10 14
xzyzyzyzqx
1 2
2 4
3 5
4 5
2 6
6 8
6 5
2 10
3 9
10 9
4 6
1 10
2 8
3 7


Output

4

Note

In the first sample, the path with largest value is 1 → 3 → 4 → 5. The value is 3 because the letter 'a' appears 3 times.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict as dd, deque

input=raw_input
range=xrange

n,m = map(int,input().split())
s = [ord(c)-ord('a') for c in input()]
E = dd(list)
P = dd(list)
C = [0]*n
for i in range(n):
    E[i] = []
for _ in range(m):
    a,b = map(int,input().split())
    E[a-1].append(b-1)
    P[b-1].append(a-1)
    C[a-1] += 1

leafs = [v for v in E if len(E[v])==0]

used = [0]*n
DP = [[0]*27 for _ in range(n)]

for i in range(n):
    DP[i][s[i]] += 1

Q = deque()
for leaf in leafs:
    Q.append(leaf)

while Q:
    node = Q.pop()
    #print(node)
    V = []
    for i in range(27):
        mx = 0
        for ch in E[node]:
            mx = max(mx,DP[ch][i])
        DP[node][i] += mx
    if used[leaf]: continue

    #for dp in DP:
    #    print(dp[:6])
    
    for pa in P[node]:
        C[pa] -= 1
        if C[pa] == 0:
            Q.append(pa)

if any(C):
    print(-1)
else:
    print(max(max(dp) for dp in DP))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque, defaultdict

class Dsubstringbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)
        self.m = params.get('m', 4)
        self.seed = params.get('seed', None)
        if self.seed is not None:
            random.seed(self.seed)
    
    def case_generator(self):
        n = self.n
        m = self.m
        
        # 生成每个节点的字母
        letters = [chr(random.randint(ord('a'), ord('z'))) for _ in range(n)]
        s = ''.join(letters)
        
        # 生成m条有向边
        edges = []
        for _ in range(m):
            x = random.randint(1, n)
            y = random.randint(1, n)
            edges.append( (x, y) )
        
        # 检查图中是否存在环
        edges_0based = [ (x-1, y-1) for x, y in edges ]
        if self.has_cycle(n, edges_0based):
            correct_answer = -1
        else:
            # 计算最长路径的值
            correct_answer = self.compute_max_path(n, edges_0based, letters)
        
        # 将edges转换为1-based的字符串表示
        edges_str = [[str(x), str(y)] for x, y in edges]
        
        case = {
            'n': n,
            'm': m,
            's': s,
            'edges': edges_str,
            'correct_answer': correct_answer
        }
        
        return case
    
    def has_cycle(self, n, edges):
        adj = [[] for _ in range(n)]
        in_degree = [0] * n
        for u, v in edges:
            adj[u].append(v)
            in_degree[v] += 1
        
        queue = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)
        
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        return count != n
    
    def compute_max_path(self, n, edges, letters):
        E = defaultdict(list)
        P = defaultdict(list)
        C = [0] * n
        
        for u, v in edges:
            E[u].append(v)
            P[v].append(u)
            C[u] += 1
        
        leafs = [u for u in E if len(E[u]) == 0]
        
        if not leafs:
            return -1
        
        DP = [ [0]*27 for _ in range(n) ]
        for i in range(n):
            c = ord(letters[i]) - ord('a')
            DP[i][c] = 1
        
        Q = deque(leafs)
        used = [False] * n
        
        while Q:
            u = Q.popleft()
            if used[u]:
                continue
            used[u] = True
            
            for c in range(27):
                max_val = 0
                for v in E[u]:
                    if DP[v][c] > max_val:
                        max_val = DP[v][c]
                DP[u][c] += max_val
            
            for v in P[u]:
                C[v] -= 1
                if C[v] == 0:
                    Q.append(v)
        
        if any(c > 0 for c in C):
            return -1
        else:
            max_value = max(max(row) for row in DP)
            return max_value
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        s = question_case['s']
        edges = question_case['edges']
        
        edge_lines = '\n'.join([f"{x} {y}" for x, y in edges])
        
        prompt = f"""You are given a graph with {n} nodes and {m} directed edges. Each node is assigned a lowercase letter, as follows: {s}.

The directed edges are:
{edge_lines}

The value of a path is defined as the highest frequency of any single letter along that path. For example, a path with letters 'abaca' has a value of 3 because 'a' appears three times.

Your task is to determine the maximum possible value of any path in this graph. If the graph contains a cycle, making the path infinitely long, output -1. Otherwise, output the maximum value.

Please provide your answer within [answer] tags.
"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        start = output.rfind('[answer]')
        if start == -1:
            return None
        end = output.find('[/answer]', start)
        if end == -1:
            return None
        answer_str = output[start+8:end].strip()
        if not answer_str:
            return None
        try:
            return int(answer_str)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_answer = identity['correct_answer']
        if solution is None:
            return False
        return solution == correct_answer
