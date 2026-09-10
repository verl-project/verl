"""# 

### 谜题描述
New Year is coming in Tree World! In this world, as the name implies, there are n cities connected by n - 1 roads, and for any two distinct cities there always exists a path between them. The cities are numbered by integers from 1 to n, and the roads are numbered by integers from 1 to n - 1. Let's define d(u, v) as total length of roads on the path between city u and city v.

As an annual event, people in Tree World repairs exactly one road per year. As a result, the length of one road decreases. It is already known that in the i-th year, the length of the ri-th road is going to become wi, which is shorter than its length before. Assume that the current year is year 1.

Three Santas are planning to give presents annually to all the children in Tree World. In order to do that, they need some preparation, so they are going to choose three distinct cities c1, c2, c3 and make exactly one warehouse in each city. The k-th (1 ≤ k ≤ 3) Santa will take charge of the warehouse in city ck.

It is really boring for the three Santas to keep a warehouse alone. So, they decided to build an only-for-Santa network! The cost needed to build this network equals to d(c1, c2) + d(c2, c3) + d(c3, c1) dollars. Santas are too busy to find the best place, so they decided to choose c1, c2, c3 randomly uniformly over all triples of distinct numbers from 1 to n. Santas would like to know the expected value of the cost needed to build the network.

However, as mentioned, each year, the length of exactly one road decreases. So, the Santas want to calculate the expected after each length change. Help them to calculate the value.

Input

The first line contains an integer n (3 ≤ n ≤ 105) — the number of cities in Tree World.

Next n - 1 lines describe the roads. The i-th line of them (1 ≤ i ≤ n - 1) contains three space-separated integers ai, bi, li (1 ≤ ai, bi ≤ n, ai ≠ bi, 1 ≤ li ≤ 103), denoting that the i-th road connects cities ai and bi, and the length of i-th road is li.

The next line contains an integer q (1 ≤ q ≤ 105) — the number of road length changes.

Next q lines describe the length changes. The j-th line of them (1 ≤ j ≤ q) contains two space-separated integers rj, wj (1 ≤ rj ≤ n - 1, 1 ≤ wj ≤ 103). It means that in the j-th repair, the length of the rj-th road becomes wj. It is guaranteed that wj is smaller than the current length of the rj-th road. The same road can be repaired several times.

Output

Output q numbers. For each given change, print a line containing the expected cost needed to build the network in Tree World. The answer will be considered correct if its absolute and relative error doesn't exceed 10 - 6.

Examples

Input

3
2 3 5
1 3 3
5
1 4
2 2
1 2
2 1
1 1


Output

14.0000000000
12.0000000000
8.0000000000
6.0000000000
4.0000000000


Input

6
1 5 3
5 3 2
6 1 7
1 4 4
5 2 3
5
1 2
2 1
3 5
4 1
5 2


Output

19.6000000000
18.6000000000
16.6000000000
13.6000000000
12.6000000000

Note

Consider the first sample. There are 6 triples: (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1). Because n = 3, the cost needed to build the network is always d(1, 2) + d(2, 3) + d(3, 1) for all the triples. So, the expected cost equals to d(1, 2) + d(2, 3) + d(3, 1).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

cost = []

def readarray(): return map(int, raw_input().split(' '))

n = int(raw_input())
graph = [[] for i in xrange(n)]

for i in xrange(n - 1):
	u, v, c = readarray()
	u, v = u - 1, v - 1
	cost.append(c)
	graph[u].append((v, i))
	graph[v].append((u, i))
	

order = []
used = set()
q = [0] * (n + n)

qh = qt = 0


used.add(0)
qh += 1

while qt < qh:
	v = q[qt]
	qt += 1
	
	order.append(v)
	
	for (to, e) in graph[v]:
		if to in used:
			continue
		used.add(to)
		q[qh] = to
		qh += 1
		
order.reverse()
		
sz = [0 for x in xrange(n)]

for v in order:
	sz[v] = 1
	for (to, e) in graph[v]:
		sz[v] += sz[to]


distanceSum = 0.0
edgeMult = [0] * n

for v in xrange(n):
	for (to, e) in graph[v]:
		x = min(sz[v], sz[to])
		edgeMult[e] = x
		distanceSum += 1.0 * cost[e] * x * (n - x)
		
distanceSum /= 2.0

queryCnt = int(raw_input())

ans = []

for i in xrange(queryCnt):
	x, y = readarray()
	x -= 1
	
	distanceSum -= 1.0 * cost[x] * edgeMult[x] * (n - edgeMult[x])
	cost[x] = y
	distanceSum += 1.0 * cost[x] * edgeMult[x] * (n - edgeMult[x])
	
	ans.append('%.10lf' % (distanceSum / n / (n - 1) * 6.0))

print('\n'.join(ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from collections import deque
from math import isclose

class Dnewyearsantanetworkbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.n = params.get('n', 3)
        self.q = params.get('q', 5)
        self.max_initial_length = params.get('max_initial_length', 1000)

    def case_generator(self):
        n = self.n
        
        # Generate a tree structure with n nodes (0-based)
        parents = [0] * n
        edges = []
        for i in range(1, n):
            parents[i] = random.randint(0, i-1)
            li = random.randint(1, self.max_initial_length)
            edges.append((parents[i], i, li))
        
        # Convert edges to 1-based
        converted_edges = [(u + 1, v + 1, li) for u, v, li in edges]
        
        # Generate queries
        current_li = [li for _, _, li in edges]
        queries = []
        available_edges = list(range(len(edges)))
        for _ in range(self.q):
            if not available_edges:
                break  # Cannot generate more queries, but according to input constraints, this shouldn't happen
            # Retry to find an edge that can be reduced
            found = False
            for _ in range(100):
                r_index = random.choice(available_edges)
                if current_li[r_index] > 1:
                    found = True
                    break
            if not found:
                r_index = available_edges[0]
                if current_li[r_index] <= 1:
                    break
            
            current_length = current_li[r_index]
            wj = random.randint(1, current_length - 1) if current_length > 1 else 0
            if wj <= 0:
                wj = 0
            queries.append((r_index + 1, wj))
            current_li[r_index] = wj
        
        case = {
            'n': n,
            'edges': converted_edges,
            'queries': queries
        }
        return case

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        edges = question_case['edges']
        queries = question_case['queries']
        prompt = (
            "You are a programming expert in Tree World. The problem involves calculating the expected cost after each road repair in a tree structure.\n"
            "The tree has n cities connected by n-1 roads. Each year, the length of one road is reduced. After each modification, output the expected cost of building the Santa network, which is the average of the sum of pairwise distances for all possible triplets of distinct cities.\n"
            "\n"
            "Input format:\n"
            "- First line: integer n (3 ≤ n ≤ 1e5)\n"
            "- Next n-1 lines: three integers ai, bi, li (describing a road)\n"
            "- Next line: integer q (number of modifications)\n"
            "- Next q lines: two integers r_j, w_j (road number and new length)\n"
            "\n"
            "Output format:\n"
            "- After each modification, print the expected cost with 10 decimal places.\n"
            "\n"
            "Input data for this problem:\n"
            f"{n}\n"
        )
        for ai, bi, li in edges:
            prompt += f"{ai} {bi} {li}\n"
        prompt += f"{len(queries)}\n"
        for rj, wj in queries:
            prompt += f"{rj} {wj}\n"
        prompt += (
            "\n"
            "Please output the expected cost after each modification, each on a new line with exactly 10 decimal places.\n"
            "Enclose your answer within [answer] and [/answer] tags. Example:\n"
            "[answer]\n"
            "14.0000000000\n"
            "12.0000000000\n"
            "...\n"
            "[/answer]\n"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        solutions = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                solutions.append(float(line))
            except ValueError:
                continue
        return solutions if solutions else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        
        n = identity['n']
        edges = identity['edges']
        queries = identity['queries']
        
        try:
            correct_output = cls.calculate_expected_output(n, edges, queries)
        except Exception as e:
            print(f"Error calculating correct output: {e}")
            return False
        
        if len(solution) != len(correct_output):
            return False
        
        for s, c in zip(solution, correct_output):
            if not cls.is_close(s, c):
                return False
        return True

    @staticmethod
    def is_close(a, b, rel_tol=1e-6, abs_tol=1e-10):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    @staticmethod
    def calculate_expected_output(n, edges, queries):
        # Convert edges to 0-based
        converted_edges = []
        for ai, bi, li in edges:
            u = ai - 1
            v = bi - 1
            converted_edges.append((u, v, li))
        
        # Build the graph and cost array
        cost = [li for u, v, li in converted_edges]
        graph = [[] for _ in range(n)]
        for i, (u, v, li) in enumerate(converted_edges):
            graph[u].append((v, i))
            graph[v].append((u, i))
        
        # Compute order via BFS and reverse
        order = []
        used = set([0])
        q = deque([0])
        while q:
            v = q.popleft()
            order.append(v)
            for (to, e) in graph[v]:
                if to not in used:
                    used.add(to)
                    q.append(to)
        order.reverse()
        
        # Compute sz
        sz = [0] * n
        for v in order:
            sz[v] = 1
            for (to, e) in graph[v]:
                if sz[to] > 0:  # Only consider children
                    sz[v] += sz[to]
        
        # Calculate edgeMult and initial distanceSum
        edgeMult = [0] * len(cost)
        distanceSum = 0.0
        for v in range(n):
            for (to, e) in graph[v]:
                x = min(sz[v], sz[to])
                edgeMult[e] = x * (n - x)
                distanceSum += cost[e] * x * (n - x)
        distanceSum /= 2.0
        
        # Process queries
        ans = []
        for rj, wj in queries:
            e_idx = rj - 1
            old_cost = cost[e_idx]
            distanceSum -= old_cost * edgeMult[e_idx]
            cost[e_idx] = wj
            distanceSum += cost[e_idx] * edgeMult[e_idx]
            expected = distanceSum * 6.0 / (n * (n - 1)) if n >= 2 else 0.0
            ans.append(expected)
        
        return ans
