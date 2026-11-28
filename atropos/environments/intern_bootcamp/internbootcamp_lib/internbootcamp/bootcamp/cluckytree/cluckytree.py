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
class DSU(object):
    def __init__(self, size=0):
        self.T = size * [-1]
    def find(self, a):
        if self.T[a] < 0:
            return a
        else:
            self.T[a] = self.find(self.T[a])
            return self.T[a]
    def merge(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if (self.T[a] > self.T[b]):
            c = a
            a = b
            b = c
        self.T[a] += self.T[b]
        self.T[b] = a

def check_lucky(n):
    while n > 0:
        digit = n % 10
        n = n / 10
        if digit != 4 and digit != 7:
            return False
    return True

n = int(raw_input())
s = DSU(n)
for _ in range(n - 1):
    node1, node2, weight = map(int, raw_input().split(' '))
    node1 -= 1
    node2 -= 1
    if not check_lucky(weight):
        s.merge(node1, node2)
res = 0
for power in s.T:
    if power < 0:
        power = - power
        res += power * (n - power) * (n - power - 1)

print res
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def is_lucky(w):
    while w > 0:
        digit = w % 10
        if digit != 4 and digit != 7:
            return False
        w = w // 10
    return True

class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]

class Cluckytreebootcamp(Basebootcamp):
    def __init__(self, **params):
        pass

    def case_generator(self):
        n = random.randint(4, 100)
        edges = []
        for _ in range(n - 1):
            u = random.randint(1, n)
            v = random.randint(1, n)
            while u == v:
                v = random.randint(1, n)
            wi = random.randint(1, 10**9)
            if random.choice([True, False]):
                digits = random.choices(['4', '7'], k=random.randint(1, 4))
                wi = int(''.join(digits))
            else:
                while is_lucky(wi):
                    wi = random.randint(1, 10**9)
            edges.append((u, v, wi))
        return {
            'n': n,
            'edges': edges
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        edges = question_case['edges']
        edges_str = '\n'.join(f"{u} {v} {w}" for u, v, w in edges)
        prompt = f"""Petya loves lucky numbers. Lucky numbers are those composed solely of the digits 4 and 7. Given a tree with {n} vertices and weighted edges, each edge's weight is a positive integer. An edge is considered lucky if its weight is a lucky number. You need to find the number of ordered triples (i,j,k) where i, j, k are distinct, and both paths from i to j and from i to k include at least one lucky edge.

The tree is defined as follows:
n = {n}
Edges:
{edges_str}

Please calculate the number of such triples and provide your answer within [answer] and [/answer].
The answer should be an integer placed within [answer] and [/answer] tags as shown below:
[answer]1234[/answer]
"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        edges = identity['edges']
        dsu = DSU(n)
        for u, v, w in edges:
            if not is_lucky(w):
                dsu.union(u-1, v-1)
        size = {}
        for i in range(n):
            root = dsu.find(i)
            if root not in size:
                size[root] = 0
            size[root] += 1
        total = 0
        for s in size.values():
            total += s * (n - s) * (n - s - 1)
        return solution == total
