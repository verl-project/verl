"""# 

### 谜题描述
There are n cells, numbered 1,2,..., n from left to right. You have to place a robot at any cell initially. The robot must make exactly k moves.

In one move, the robot must move one cell to the left or right, provided that it doesn't move out of bounds. In other words, if the robot was in the cell i, it must move to either the cell i-1 or the cell i+1, as long as it lies between 1 and n (endpoints inclusive). The cells, in the order they are visited (including the cell the robot is placed), together make a good path.

Each cell i has a value a_i associated with it. Let c_0, c_1, ..., c_k be the sequence of cells in a good path in the order they are visited (c_0 is the cell robot is initially placed, c_1 is the cell where the robot is after its first move, and so on; more formally, c_i is the cell that the robot is at after i moves). Then the value of the path is calculated as a_{c_0} + a_{c_1} + ... + a_{c_k}.

Your task is to calculate the sum of values over all possible good paths. Since this number can be very large, output it modulo 10^9 + 7. Two good paths are considered different if the starting cell differs or there exists an integer i ∈ [1, k] such that the current cell of the robot after exactly i moves is different in those paths.

You must process q updates to a and print the updated sum each time. Each update changes the value of exactly one cell. See the input format and the sample input-output for more details.

Input

The first line of the input contains three space-separated integers n, k and q (2 ≤ n ≤ 5000; 1 ≤ k ≤ 5000; 1 ≤ q ≤ 2 ⋅ 10^5).

The second line of the input contains n integers a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^9).

q lines follow. Each line contains two space-separated integers i and x (1 ≤ i ≤ n; 1 ≤ x ≤ 10^9) indicating that you must change the value of a_i to x.

Output

Print q integers. The i-th integer should be the sum of values over all good paths after the first i updates are performed. Since the answers may be large, print them modulo 10^9 + 7.

Examples

Input


5 1 5
3 5 1 4 2
1 9
2 4
3 6
4 6
5 2


Output


62
58
78
86
86


Input


5 2 5
3 5 1 4 2
1 9
2 4
3 6
4 6
5 2


Output


157
147
207
227
227


Input


4 40 6
92 21 82 46
3 56
1 72
4 28
1 97
2 49
2 88


Output


239185261
666314041
50729936
516818968
766409450
756910476

Note

In the first example, the good paths are (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4).

Initially the values of a are [3, 5, 1, 4, 2]. After the first update, they become [9, 5, 1, 4, 2]. After the second update, they become [9, 4, 1, 4, 2], and so on.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function

import os
import sys
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


def main():
    pass


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

from collections import defaultdict

mod=10**9+7
n,k,q=map(int,input().split())
a=list(map(int,input().split()))
cnt=defaultdict(lambda: 0)
dp=[[0 for j in range(k+1)] for i in range(n+2)]
for i in range(1,n+1):
    dp[i][0]=1
for j in range(1,k+1):
    for i in range(1,n+1):
        dp[i][j]=(dp[i-1][j-1]+dp[i+1][j-1])%mod
for i in range(1,n+1):
    for j in range(k+1):
        cnt[i]+=dp[i][j]*dp[i][k-j]%mod
ans=0
for i in range(n):
    ans+=cnt[i+1]*a[i]%mod
for _ in range(q):
    i,x=map(int,input().split())
    ans=(ans+(x-a[i-1])*cnt[i])%mod
    a[i-1]=x
    print(ans%mod)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Dsumofpathsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)
        self.k = params.get('k', 1)
        self.q = params.get('q', 5)
        self.seed = params.get('seed', None)
        self.rng = random.Random(self.seed)
        self.min_a = params.get('min_a', 1)
        self.max_a = params.get('max_a', 10**9)

        self.n = min(max(self.n, 2), 5000)
        self.k = min(max(self.k, 1), 5000)
        self.q = min(max(self.q, 1), 2*10**5)

    def case_generator(self):
        a = [self.rng.randint(self.min_a, self.max_a) for _ in range(self.n)]
        updates = []
        for _ in range(self.q):
            i = self.rng.randint(1, self.n)
            x = self.rng.randint(self.min_a, self.max_a)
            updates.append((i, x))
        return {
            'n': self.n,
            'k': self.k,
            'q': self.q,
            'a': a,
            'updates': updates
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        q = question_case['q']
        a = question_case['a']
        updates = question_case['updates']
        
        problem = f"""You are a programming assistant tasked with solving a dynamic path sum problem. The problem is as follows:

There are {n} cells arranged in a line, numbered from 1 to {n}. Each cell i has an initial value a_i. A robot starts at any cell and makes exactly {k} moves. Each move must be to the left or right, and cannot go out of bounds. The value of a good path is the sum of the values of all cells visited (including the starting cell). Your task is to compute the sum of values over all possible good paths, modulo 1e9+7.

After the initial setup, there are {q} updates. Each update changes the value of a particular cell. After each update, you must output the new sum modulo 1e9+7.

Initial parameters:
- n = {n}
- k = {k}
- q = {q}
- Initial a array: {a} (cell 1 to {n})

Updates (each line is the i-th update changing cell i to x):\n"""
        
        for idx, (i, x) in enumerate(updates, 1):
            problem += f"Update {idx}: Change cell {i} to {x}\n"
        
        problem += """
Your task is to compute the sum after each update and output all results as integers, each on a new line. Place your final answer within [answer] and [/answer] tags. For example:

[answer]
123
456
...
[/answer]

Ensure all numbers are present and in the correct order. Each number must be the computed sum modulo 1e9+7 after the corresponding update."""
        return problem

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        numbers = re.findall(r'\b\d+\b', last_block)
        try:
            solution = list(map(int, numbers))
        except ValueError:
            return None
        return solution

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list) or len(solution) != identity['q']:
            return False
        
        n = identity['n']
        k = identity['k']
        a = identity['a'].copy()
        updates = identity['updates']

        # Optimized path calculation
        dp = [[0]*(k+1) for _ in range(n+2)]
        for i in range(1, n+1):
            dp[i][0] = 1
        for j in range(1, k+1):
            for i in range(1, n+1):
                left = dp[i-1][j-1] if i > 1 else 0
                right = dp[i+1][j-1] if i < n else 0
                dp[i][j] = (left + right) % MOD

        cnt = defaultdict(int)
        for i in range(1, n+1):
            total = 0
            for j in range(k+1):
                total = (total + dp[i][j] * dp[i][k-j]) % MOD
            cnt[i] = total

        ans = 0
        for idx in range(n):
            ans = (ans + a[idx] * cnt[idx+1]) % MOD
        
        correct = []
        for (i_update, x) in updates:
            idx_array = i_update - 1
            delta = (x - a[idx_array]) * cnt[i_update]
            ans = (ans + delta) % MOD
            a[idx_array] = x
            correct.append(ans % MOD)
        
        return solution == correct
