"""# 

### 谜题描述
The main characters have been omitted to be short.

You are given a directed unweighted graph without loops with n vertexes and a path in it (that path is not necessary simple) given by a sequence p_1, p_2, …, p_m of m vertexes; for each 1 ≤ i < m there is an arc from p_i to p_{i+1}.

Define the sequence v_1, v_2, …, v_k of k vertexes as good, if v is a subsequence of p, v_1 = p_1, v_k = p_m, and p is one of the shortest paths passing through the vertexes v_1, …, v_k in that order.

A sequence a is a subsequence of a sequence b if a can be obtained from b by deletion of several (possibly, zero or all) elements. It is obvious that the sequence p is good but your task is to find the shortest good subsequence.

If there are multiple shortest good subsequences, output any of them.

Input

The first line contains a single integer n (2 ≤ n ≤ 100) — the number of vertexes in a graph. 

The next n lines define the graph by an adjacency matrix: the j-th character in the i-st line is equal to 1 if there is an arc from vertex i to the vertex j else it is equal to 0. It is guaranteed that the graph doesn't contain loops.

The next line contains a single integer m (2 ≤ m ≤ 10^6) — the number of vertexes in the path. 

The next line contains m integers p_1, p_2, …, p_m (1 ≤ p_i ≤ n) — the sequence of vertexes in the path. It is guaranteed that for any 1 ≤ i < m there is an arc from p_i to p_{i+1}.

Output

In the first line output a single integer k (2 ≤ k ≤ m) — the length of the shortest good subsequence. In the second line output k integers v_1, …, v_k (1 ≤ v_i ≤ n) — the vertexes in the subsequence. If there are multiple shortest subsequences, print any. Any two consecutive numbers should be distinct.

Examples

Input


4
0110
0010
0001
1000
4
1 2 3 4


Output


3
1 2 4 

Input


4
0110
0010
1001
1000
20
1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4


Output


11
1 2 4 2 4 2 4 2 4 2 4 

Input


3
011
101
110
7
1 2 3 1 3 2 1


Output


7
1 2 3 1 3 2 1 

Input


4
0110
0001
0001
1000
3
1 2 4


Output


2
1 4 

Note

Below you can see the graph from the first example:

<image>

The given path is passing through vertexes 1, 2, 3, 4. The sequence 1-2-4 is good because it is the subsequence of the given path, its first and the last elements are equal to the first and the last elements of the given path respectively, and the shortest path passing through vertexes 1, 2 and 4 in that order is 1-2-3-4. Note that subsequences 1-4 and 1-3-4 aren't good because in both cases the shortest path passing through the vertexes of these sequences is 1-3-4.

In the third example, the graph is full so any sequence of vertexes in which any two consecutive elements are distinct defines a path consisting of the same number of vertexes.

In the fourth example, the paths 1-2-4 and 1-3-4 are the shortest paths passing through the vertexes 1 and 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

n = int(input())
M = [[int(x) for x in input()] for _ in range(n)]

def floyd_warshall(n, dist):
    dist = [[d if d > 0 else 1000000000 for d in line] for line in dist]
    for i in range(n):
        dist[i][i] = 0
    
    pred = [[-1] * n for _ in range(n)]

    for u in range(n):
        for v in range(n):
            if dist[u][v]:
                pred[u][v] = u

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]
    return dist, pred


dist, pred = floyd_warshall(n, [list(a) for a in M])

m = int(input())
P = [int(x) - 1 for x in input().split()]


DP = [-1]*m
DP[0] = -2

pairs = []
for i in range(m):
    if DP[i] != -1:
        d = dist[P[i]]
        j = i + 1
        while j < m and d[P[j]] == j - i:
            if DP[j] == -1:
                DP[j] = i
            j += 1

k = m - 1
ans = []
while k >= 0:
    ans.append(k)
    k = DP[k]

print len(ans)
print ' '.join(str(P[x] + 1) for x in ans[::-1])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cannasvyatoslavandmapsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.params.setdefault('n_min', 2)
        self.params.setdefault('n_max', 10)
        self.params.setdefault('m_min', 2)
        self.params.setdefault('m_max', 20)

    def case_generator(self):
        n = random.randint(self.params['n_min'], self.params['n_max'])
        m = random.randint(self.params['m_min'], self.params['m_max'])
        
        # Generate valid path with guaranteed edges
        p = []
        current = random.randint(1, n)
        p.append(current)
        adj_matrix = [[0]*n for _ in range(n)]
        
        for _ in range(m-1):
            available = [v for v in range(1, n+1) if v != current]
            next_v = random.choice(available)
            adj_matrix[current-1][next_v-1] = 1  # Mark existing edge
            p.append(next_v)
            current = next_v

        # Fill remaining edges (avoid self loops)
        for u in range(n):
            for v in range(n):
                if u != v and adj_matrix[u][v] == 0:
                    adj_matrix[u][v] = random.choice([0, 1])

        return {
            'n': n,
            'adj_matrix': [''.join(map(str, row)) for row in adj_matrix],
            'm': m,
            'p': p
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        adj_matrix = question_case['adj_matrix']
        m = question_case['m']
        p_str = ' '.join(map(str, question_case['p']))
        
        return f"""You are given a directed graph with {n} vertices and a path. Find the shortest good subsequence.

Graph adjacency matrix:
"""+'\n'.join(adj_matrix)+f"""

Path ({m} vertices):
{p_str}

A good subsequence must:
1. Be a subsequence starting and ending with path's first/last vertex
2. Original path must be the shortest path for this subsequence

Output format:
[answer]
k
v1 v2 ... vk
[/answer]"""

    @staticmethod
    def extract_output(output):
        match = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not match:
            return None
        last_answer = match[-1].strip()
        numbers = list(map(int, re.findall(r'\d+', last_answer)))
        if len(numbers) < 2:
            return None
        return numbers[1:]

    @classmethod
    def _verify_correction(cls, solution, identity):
        # Validate basic format
        if len(solution) < 2 or solution[0] != identity['p'][0] or solution[-1] != identity['p'][-1]:
            return False
        
        # Check subsequence condition
        p_iter = iter(identity['p'])
        try:
            for v in solution:
                while next(p_iter) != v:
                    pass
        except StopIteration:
            return False

        # Validate path is shortest
        n = identity['n']
        adj = [[int(c) for c in row] for row in identity['adj_matrix']]
        
        # Build distance matrix
        dist = [[float('inf')]*n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            for j in range(n):
                if adj[i][j]:
                    dist[i][j] = 1
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        # Calculate required path length
        total = 0
        for i in range(len(solution)-1):
            u = solution[i]-1
            v = solution[i+1]-1
            if dist[u][v] == float('inf'):
                return False
            total += dist[u][v]
        
        return total == (identity['m'] - 1)
