"""# 

### 谜题描述
Uncle Bogdan is in captain Flint's crew for a long time and sometimes gets nostalgic for his homeland. Today he told you how his country introduced a happiness index.

There are n cities and n−1 undirected roads connecting pairs of cities. Citizens of any city can reach any other city traveling by these roads. Cities are numbered from 1 to n and the city 1 is a capital. In other words, the country has a tree structure.

There are m citizens living in the country. A p_i people live in the i-th city but all of them are working in the capital. At evening all citizens return to their home cities using the shortest paths. 

Every person has its own mood: somebody leaves his workplace in good mood but somebody are already in bad mood. Moreover any person can ruin his mood on the way to the hometown. If person is in bad mood he won't improve it.

Happiness detectors are installed in each city to monitor the happiness of each person who visits the city. The detector in the i-th city calculates a happiness index h_i as the number of people in good mood minus the number of people in bad mood. Let's say for the simplicity that mood of a person doesn't change inside the city.

Happiness detector is still in development, so there is a probability of a mistake in judging a person's happiness. One late evening, when all citizens successfully returned home, the government asked uncle Bogdan (the best programmer of the country) to check the correctness of the collected happiness indexes.

Uncle Bogdan successfully solved the problem. Can you do the same?

More formally, You need to check: \"Is it possible that, after all people return home, for each city i the happiness index will be equal exactly to h_i\".

Input

The first line contains a single integer t (1 ≤ t ≤ 10000) — the number of test cases.

The first line of each test case contains two integers n and m (1 ≤ n ≤ 10^5; 0 ≤ m ≤ 10^9) — the number of cities and citizens.

The second line of each test case contains n integers p_1, p_2, …, p_{n} (0 ≤ p_i ≤ m; p_1 + p_2 + … + p_{n} = m), where p_i is the number of people living in the i-th city.

The third line contains n integers h_1, h_2, …, h_{n} (-10^9 ≤ h_i ≤ 10^9), where h_i is the calculated happiness index of the i-th city.

Next n − 1 lines contain description of the roads, one per line. Each line contains two integers x_i and y_i (1 ≤ x_i, y_i ≤ n; x_i ≠ y_i), where x_i and y_i are cities connected by the i-th road.

It's guaranteed that the sum of n from all test cases doesn't exceed 2 ⋅ 10^5.

Output

For each test case, print YES, if the collected data is correct, or NO — otherwise. You can print characters in YES or NO in any case.

Examples

Input


2
7 4
1 0 1 1 0 1 0
4 0 0 -1 0 -1 0
1 2
1 3
1 4
3 5
3 6
3 7
5 11
1 2 5 2 1
-11 -2 -6 -2 -1
1 2
1 3
1 4
3 5


Output


YES
YES


Input


2
4 4
1 1 1 1
4 1 -3 -1
1 2
1 3
1 4
3 13
3 3 7
13 1 4
1 2
1 3


Output


NO
NO

Note

Let's look at the first test case of the first sample: 

<image>

At first, all citizens are in the capital. Let's describe one of possible scenarios: 

  * a person from city 1: he lives in the capital and is in good mood; 
  * a person from city 4: he visited cities 1 and 4, his mood was ruined between cities 1 and 4; 
  * a person from city 3: he visited cities 1 and 3 in good mood; 
  * a person from city 6: he visited cities 1, 3 and 6, his mood was ruined between cities 1 and 3; 

In total, 
  * h_1 = 4 - 0 = 4, 
  * h_2 = 0, 
  * h_3 = 1 - 1 = 0, 
  * h_4 = 0 - 1 = -1, 
  * h_5 = 0, 
  * h_6 = 0 - 1 = -1, 
  * h_7 = 0. 



The second case of the first test: 

<image>

All people have already started in bad mood in the capital — this is the only possible scenario.

The first case of the second test: 

<image>

The second case of the second test: 

<image>

It can be proven that there is no way to achieve given happiness indexes in both cases of the second test. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
from __future__ import division, print_function

import os
import sys
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


def dfs(graph, p, h, start=0):
    n = len(graph)

    dp = [[0, 0] for _ in range(n)]
    visited, finished = [False] * n, [False] * n

    stack = [start]
    while stack:
        start = stack[-1]

        # push unvisited children into stack
        if not visited[start]:
            visited[start] = True
            for child in graph[start]:
                if not visited[child]:
                    stack.append(child)

        else:
            stack.pop()

            # base case
            dp[start][0] = p[start]
            dp[start][1] = 0

            # update with finished children
            for child in graph[start]:
                if finished[child]:
                    dp[start][0] += dp[child][0]
                    dp[start][1] += dp[child][1]

            if dp[start][1] - dp[start][0] <= h[start] <= dp[start][1] + dp[start][0] and h[start] % 2 == (dp[start][1] - dp[start][0]) % 2:
                v = (h[start] - (dp[start][1] - dp[start][0])) // 2
                dp[start][1] += v
                dp[start][0] -= v
            else:
                return False

            finished[start] = True

    return True
    #return visited, dp


def main():
    t = int(input())

    for _ in range(t):
        n, m = map(int, input().split())
        p = [int(pi) for pi in input().split()]
        h = [int(hi) for hi in input().split()]

        graph = [[] for _ in range(n)]

        for _ in range(n - 1):
            x, y = map(int, input().split())
            graph[x - 1].append(y - 1)
            graph[y - 1].append(x - 1)

        tree = [[] for _ in range(n)]
        stack = [0]
        visited = [False] * n

        while stack:
            curr = stack.pop()
            visited[curr] = True

            for child in graph[curr]:
                if not visited[child]:
                    tree[curr].append(child)
                    stack.append(child)

        print('YES' if dfs(tree, p, h) else 'NO')


# region fastio

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = \"x\" in file.mode or \"r\" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b\"\n\") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))
        self.read = lambda: self.buffer.read().decode(\"ascii\")
        self.readline = lambda: self.buffer.readline().decode(\"ascii\")


def print(*args, **kwargs):
    \"\"\"Prints the values to a stream, or to sys.stdout by default.\"\"\"
    sep, file = kwargs.pop(\"sep\", \" \"), kwargs.pop(\"file\", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop(\"end\", \"\n\"))
    if kwargs.pop(\"flush\", False):
        file.flush()


if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip(\"\r\n\")

# endregion

if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque
from bootcamp import Basebootcamp

def generate_tree_edges(n):
    if n == 1:
        return []
    parents = [0] * (n + 1)  # 1-based index
    for i in range(2, n + 1):
        parents[i] = random.randint(1, i - 1)
    return [(parents[i], i) for i in range(2, n + 1)]

def generate_p(n, m):
    if m == 0:
        return [0] * n
    p = []
    remaining = m
    for _ in range(n - 1):
        val = random.randint(0, remaining)
        p.append(val)
        remaining -= val
    p.append(remaining)
    return p

def build_tree_and_parents(n, edges):
    adj = [[] for _ in range(n + 1)]  # 1-based
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    parent = [0] * (n + 1)
    visited = [False] * (n + 1)
    q = deque([1])
    visited[1] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                q.append(v)
    return parent

class Cunclebogdanandcountryhappinessbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=5, m_min=1, m_max=100):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        edges = generate_tree_edges(n)
        parent = build_tree_and_parents(n, edges)
        
        # Generate p with sum m
        p = generate_p(n, m)
        
        # Generate valid h by simulating people's mood changes
        good = [0] * (n + 1)  # 1-based
        bad = [0] * (n + 1)
        
        # For each city, distribute people
        for city in range(1, n + 1):
            num_people = p[city - 1]
            if num_people == 0:
                continue
            
            # Path from capital (1) to current city
            path = []
            current = city
            while current != 1:
                path.append(current)
                current = parent[current]
            path.append(1)
            path = path[::-1]  # reverse to get path from 1 to city
            
            for _ in range(num_people):
                # Randomly choose when the mood is ruined (None means never)
                ruin_step = random.randint(0, len(path)-1)
                
                # Update mood for each city in path
                for step in range(len(path)):
                    current_city = path[step]
                    if step < ruin_step:
                        good[current_city] += 1
                    else:
                        bad[current_city] += 1
        
        # Compute h_i = good[i] - bad[i]
        h = [good[i] - bad[i] for i in range(1, n + 1)]
        
        # Randomly decide to make invalid case with 50% chance
        if random.random() < 0.5:
            # Modify h to create invalid case
            idx = random.randint(0, n-1)
            h[idx] += random.choice([-1, 1])
        
        # Generate input data
        input_lines = [
            "1",
            f"{n} {m}",
            " ".join(map(str, p)),
            " ".join(map(str, h))
        ]
        input_lines.extend(f"{u} {v}" for u, v in edges)
        input_str = "\n".join(input_lines) + "\n"
        
        # Solve to get expected answer
        expected = 'YES' if self.solve_happiness(input_str) else 'NO'
        return {
            "n": n, "m": m, "p": p, "h": h,
            "edges": edges, "expected_answer": expected
        }
    
    def solve_happiness(self, input_str):
        from io import StringIO
        import sys
        old_stdin = sys.stdin
        sys.stdin = StringIO(input_str)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        main()
        
        output = sys.stdout.getvalue().strip().upper()
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        return output == 'YES'

    @staticmethod
    def prompt_func(question_case):
        edges_str = "\n".join(f"{u} {v}" for u, v in question_case["edges"])
        prompt = f"""Determine if the happiness indexes are possible. Follow the input format:

Cities: {question_case['n']}
People: {question_case['m']}
Population: {' '.join(map(str, question_case['p']))}
Cunclebogdanandcountryhappiness: {' '.join(map(str, question_case['h']))}
Edges:
{edges_str}

Output YES or NO within [answer]...[/answer]."""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip().upper()
        return answer if answer in {'YES', 'NO'} else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_answer']

# Reference solution implementation remains unchanged
def dfs(graph, p, h, start=0):
    n = len(graph)
    dp = [[0, 0] for _ in range(n)]
    visited, finished = [False]*n, [False]*n
    stack = [start]
    while stack:
        curr = stack[-1]
        if not visited[curr]:
            visited[curr] = True
            for child in graph[curr]:
                if not visited[child]:
                    stack.append(child)
        else:
            curr = stack.pop()
            dp[curr][0] = p[curr]
            dp[curr][1] = 0
            for child in graph[curr]:
                if finished[child]:
                    dp[curr][0] += dp[child][0]
                    dp[curr][1] += dp[child][1]
            lower = dp[curr][1] - dp[curr][0]
            upper = dp[curr][1] + dp[curr][0]
            if not (lower <= h[curr] <= upper and (h[curr] - lower) % 2 == 0):
                return False
            v = (h[curr] - lower) // 2
            dp[curr][1] += v
            dp[curr][0] -= v
            finished[curr] = True
    return True

def main():
    t = int(input())
    for _ in range(t):
        n, m = map(int, input().split())
        p = list(map(int, input().split()))
        h = list(map(int, input().split()))
        graph = [[] for _ in range(n)]
        for _ in range(n-1):
            x, y = map(int, input().split())
            x -= 1; y -= 1
            graph[x].append(y); graph[y].append(x)
        tree = [[] for _ in range(n)]
        visited = [False]*n
        stack = [0]
        while stack:
            curr = stack.pop()
            visited[curr] = True
            for child in graph[curr]:
                if not visited[child]:
                    tree[curr].append(child)
                    stack.append(child)
        print("YES" if dfs(tree, p, h) else "NO")
