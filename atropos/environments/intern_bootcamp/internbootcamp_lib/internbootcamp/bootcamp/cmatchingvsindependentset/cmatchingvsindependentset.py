"""# 

### 谜题描述
You are given a graph with 3 ⋅ n vertices and m edges. You are to find a matching of n edges, or an independent set of n vertices.

A set of edges is called a matching if no two edges share an endpoint.

A set of vertices is called an independent set if no two vertices are connected with an edge.

Input

The first line contains a single integer T ≥ 1 — the number of graphs you need to process. The description of T graphs follows.

The first line of description of a single graph contains two integers n and m, where 3 ⋅ n is the number of vertices, and m is the number of edges in the graph (1 ≤ n ≤ 10^{5}, 0 ≤ m ≤ 5 ⋅ 10^{5}).

Each of the next m lines contains two integers v_i and u_i (1 ≤ v_i, u_i ≤ 3 ⋅ n), meaning that there is an edge between vertices v_i and u_i.

It is guaranteed that there are no self-loops and no multiple edges in the graph.

It is guaranteed that the sum of all n over all graphs in a single test does not exceed 10^{5}, and the sum of all m over all graphs in a single test does not exceed 5 ⋅ 10^{5}.

Output

Print your answer for each of the T graphs. Output your answer for a single graph in the following format.

If you found a matching of size n, on the first line print \"Matching\" (without quotes), and on the second line print n integers — the indices of the edges in the matching. The edges are numbered from 1 to m in the input order.

If you found an independent set of size n, on the first line print \"IndSet\" (without quotes), and on the second line print n integers — the indices of the vertices in the independent set.

If there is no matching and no independent set of the specified size, print \"Impossible\" (without quotes).

You can print edges and vertices in any order.

If there are several solutions, print any. In particular, if there are both a matching of size n, and an independent set of size n, then you should print exactly one of such matchings or exactly one of such independent sets.

Example

Input


4
1 2
1 3
1 2
1 2
1 3
1 2
2 5
1 2
3 1
1 4
5 1
1 6
2 15
1 2
1 3
1 4
1 5
1 6
2 3
2 4
2 5
2 6
3 4
3 5
3 6
4 5
4 6
5 6


Output


Matching
2
IndSet
1
IndSet
2 4
Matching
1 15

Note

The first two graphs are same, and there are both a matching of size 1 and an independent set of size 1. Any of these matchings and independent sets is a correct answer.

The third graph does not have a matching of size 2, however, there is an independent set of size 2. Moreover, there is an independent set of size 5: 2 3 4 5 6. However such answer is not correct, because you are asked to find an independent set (or matching) of size exactly n.

The fourth graph does not have an independent set of size 2, but there is a matching of size 2.

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

import random
import collections
import math
import itertools
import bisect

def real_main():
    T = read_int()
    for _ in range(T):
        n, m = read_int_array()
        v = n * 3

        g = [[] for _ in range(v)]
        E = []

        for e in range(m):
            a, b = read_int_array()
            a -= 1; b -= 1
            E.append((a, b))
            g[a].append(b)
            g[b].append(a)

        used = [False] * v

        matching = []
        for e, (a, b) in enumerate(E):
            if not(used[a] or used[b]):
                matching.append(e)
                used[a] = used[b] = True
                if len(matching) == n:
                    print(\"Matching\")
                    print(' '.join(str(e+1) for e in matching))
                    break
        else:
            print(\"IndSet\")
            ind_set = [i for i in range(v) if not used[i]]
            print(' '.join(str(i + 1) for i in ind_set[:n]))


def solve():
    pass


def main():
    if False and 'PYCHARM_HOSTED' in os.environ:
        main_pycharm()
    else:
        real_main()


from copy import deepcopy
def main_pycharm():
    solution = solve

    test_inputs = None
    test_outputs = None
    judge = None
    slow_solution = None
    if solution is not None:
        if test_outputs is not None:
            test_with_answers(solution, test_inputs, test_outputs)
        if judge is not None:
            test_with_judge(solution, test_inputs, judge)
        if slow_solution is not None:
            test_with_slow_solution(solution, test_inputs, slow_solution)


def test_with_answers(solution, inputs, answers):
    total, wrong = 0, 0
    for args, test_ans in zip(inputs, answers):
        ans = solution(*deepcopy(args))
        if ans != test_ans:
            print('WRONG! ans=%s, test_ans=%s, args=%s' % (ans, test_ans, args))
            wrong += 1
        else:
            print('GOOD')
        total += 1
    print('ALL %d TESTS PASSED' % total if not wrong else '%d out of %d tests are WRONG' % (wrong, total))


def test_with_judge(solution, inputs_gen, judge):
    total, wrong = 0, 0
    for args in inputs_gen:
        ans = solution(*deepcopy(args))
        if not judge(deepcopy(ans), *deepcopy(args)):
            print('WRONG! ans=%s, args=%s' % (ans, args))
            wrong += 1
        total += 1
    print('ALL %d TESTS PASSED' % total if not wrong else '%d out of %d tests are WRONG' % (wrong, total))


def test_with_slow_solution(solution, inputs_gen, solution_slow):
    total, wrong = 0, 0
    for args in inputs_gen:
        ans = solution(*deepcopy(args))
        slow = solution_slow(*deepcopy(args))
        if ans != slow:
            print('WRONG! ans=%s, slow=%s, args=%s' % (ans, slow, args))
            wrong += 1
        total += 1
    print('ALL %d TESTS PASSED' % total if not wrong else '%d out of %d tests are WRONG' % (wrong, total))

def generate_nums(n, min, max, check_if_good=None):
    while True:
        nums = [random.randint(min, max) for _ in range(n)]
        if check_if_good is None or check_if_good(nums):
            return nums

# This mergesort can be like 7 times faster than build in sort
# (for stupid reasons)
def mergesort(A, key=lambda x: x, reverse=False):
    C = A
    A = list(range(len(A)))
    B = list(A)

    n = len(A)
    for i in range(0, n - 1, 2):
        if key(C[A[i]]) > key(C[A[i ^ 1]]):
            A[i], A[i ^ 1] = A[i ^ 1], A[i]

    width = 2
    while width < n:
        for i in range(0, n, 2 * width):
            R1, R2 = min(i + width, n), min(i + 2 * width, n)
            j, k = R1, i
            while i < R1 and j < R2:
                if key(C[A[i]]) > key(C[A[j]]):
                    B[k] = A[j]
                    j += 1
                else:
                    B[k] = A[i]
                    i += 1
                k += 1
            while i < R1:
                B[k] = A[i]
                k += 1
                i += 1
            while k < R2:
                B[k] = A[k]
                k += 1
        A, B = B, A
        width *= 2

    if reverse:
        A.reverse()
    return A

def mergesort_simple(A, reverse=False):
    C = A
    A = list(range(len(A)))
    B = list(A)

    n = len(A)
    for i in range(0, n - 1, 2):
        if C[A[i]] > C[A[i ^ 1]]:
            A[i], A[i ^ 1] = A[i ^ 1], A[i]

    width = 2
    while width < n:
        for i in range(0, n, 2 * width):
            R1, R2 = min(i + width, n), min(i + 2 * width, n)
            j, k = R1, i
            while i < R1 and j < R2:
                if C[A[i]] > C[A[j]]:
                    B[k] = A[j]
                    j += 1
                else:
                    B[k] = A[i]
                    i += 1
                k += 1
            while i < R1:
                B[k] = A[i]
                k += 1
                i += 1
            while k < R2:
                B[k] = A[k]
                k += 1
        A, B = B, A
        width *= 2

    if reverse:
        A.reverse()
    return A


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

def read():
    return input()

def read_int():
    return int(input())

def read_array(sep=None, maxsplit=-1):
    return input().split(sep, maxsplit)

def read_int_array():
    return [int(a) for a in read_array()]

# endregion

if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cmatchingvsindependentsetbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5, case_type='random'):
        self.min_n = min_n
        self.max_n = max_n
        self.case_type = case_type

    def case_generator(self):
        n_val = random.randint(self.min_n, self.max_n)
        case_type = self.case_type
        if case_type == 'random':
            case_type = random.choice(['matching', 'indset'])

        edges = []
        if case_type == 'matching':
            # Generate 3n vertices
            vertices = list(range(1, 3 * n_val + 1))
            random.shuffle(vertices)
            
            # Take 2n vertices to form edges
            selected = vertices[:2*n_val]
            edges = []
            for i in range(0, len(selected), 2):
                edges.append((selected[i], selected[i+1]))
        else:
            # Generate empty edges for guaranteed indset
            pass

        return {
            'n': n_val,
            'm': len(edges),
            'edges': edges,
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        edges = question_case['edges']
        input_lines = ["1", f"{n} {m}"] + [f"{a} {b}" for a, b in edges]
        input_str = '\n'.join(input_lines)
        
        prompt = f"""Given a graph with {3*n} vertices and {m} edges, find either:
1. A matching of {n} edges (no shared vertices)
2. An independent set of {n} vertices (no connected pairs)

Input format:
T=1
n m
followed by edge pairs

Output format:
'Matching' with edge indices (1-based) or 
'IndSet' with vertex indices (1-based)

[answer]
Matching
1 2 3
[/answer]
or
[answer]
IndSet
4 5 6
[/answer]

Input Data:
{input_str}

Your answer enclosed in [answer] tags:"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(
            r'\[answer\](.*?)\[\/answer\]',
            output,
            flags=re.DOTALL
        )
        if not matches:
            return None
        
        content = matches[-1].strip()
        if not content:
            return None

        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if len(lines) < 2:
            return None

        sol_type = lines[0].lower()
        nums = []
        for l in lines[1:]:
            nums.extend(map(int, l.split()))
        
        return {
            'type': 'Matching' if 'matching' in sol_type else 'IndSet',
            'elements': nums
        }

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or not identity:
            return False
        
        n = identity['n']
        edges = identity['edges']
        elements = solution.get('elements', [])
        sol_type = solution.get('type', '')

        # Validate element count
        if len(elements) != n:
            return False

        # Matching verification
        if sol_type == 'Matching':
            # Check valid edge indices
            if any(e < 1 or e > len(edges) for e in elements):
                return False
            
            # Check edges are disjoint
            seen = set()
            for e_idx in elements:
                a, b = edges[e_idx-1]
                if a in seen or b in seen:
                    return False
                seen.update({a, b})
            return True

        # IndSet verification
        elif sol_type == 'IndSet':
            # Check valid vertex numbers
            if any(v < 1 or v > 3*n for v in elements):
                return False
            
            # Check no edges between vertices
            vertices = set(elements)
            return not any(
                u in vertices and v in vertices
                for u, v in edges
            )

        return False
