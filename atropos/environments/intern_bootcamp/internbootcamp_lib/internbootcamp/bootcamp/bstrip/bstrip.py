"""# 

### 谜题描述
Alexandra has a paper strip with n numbers on it. Let's call them ai from left to right.

Now Alexandra wants to split it into some pieces (possibly 1). For each piece of strip, it must satisfy:

  * Each piece should contain at least l numbers.
  * The difference between the maximal and the minimal number on the piece should be at most s.



Please help Alexandra to find the minimal number of pieces meeting the condition above.

Input

The first line contains three space-separated integers n, s, l (1 ≤ n ≤ 105, 0 ≤ s ≤ 109, 1 ≤ l ≤ 105).

The second line contains n integers ai separated by spaces ( - 109 ≤ ai ≤ 109).

Output

Output the minimal number of strip pieces.

If there are no ways to split the strip, output -1.

Examples

Input

7 2 2
1 3 1 2 4 1 2


Output

3


Input

7 2 2
1 100 1 100 1 100 1


Output

-1

Note

For the first sample, we can split the strip into 3 pieces: [1, 3, 1], [2, 4], [1, 2].

For the second sample, we can't let 1 and 100 be on the same piece, so no solution exists.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#from sys import setrecursionlimit as srl
import sys
if sys.subversion[0] == \"PyPy\":
    import io, atexit
    sys.stdout = io.BytesIO()
    atexit.register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))
    
    sys.stdin = io.BytesIO(sys.stdin.read())
    input = lambda: sys.stdin.readline().rstrip()
 
RS = raw_input
RI = lambda x=int: map(x,RS().split())
RN = lambda x=int: x(RS())
''' ...................................................................... '''
from collections import deque
INF = 10**10

n,S,l = RI()
n += 1
arr = [0]+RI()

dp = [INF]*n; dp[0] = 0
big,small = deque(),deque()
start = 0

haha = deque([0])

for end in xrange(1,n):
    # Max Strip Seg possible -> (start,end]

    # add arr[end]
    while small and arr[small[-1]]>= arr[end]:
        small.pop()
    small.append(end)

    while big and arr[big[-1]]<= arr[end]:
        big.pop()
    big.append(end)
    
    # get start
    while arr[big[0]]-arr[small[0]]>S:
        if big[0]<small[0]:
            start = big.popleft()
        else:
            start = small.popleft()
            

    #pop haha till idx < start
    #choose the first val with idx >= start

    if start<=(end-l):
        while haha and haha[0]<start:   # haha stores dp's idx
            haha.popleft()
        if haha and haha[0]<=end-l:
            dp[end] = dp[haha[0]]+1

            while haha and dp[haha[-1]]>dp[end]:
                haha.pop()
            haha.append(end)

        
print (-1 if dp[-1]>n else dp[-1])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Bstripbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, min_s=0, max_s=10, min_l=1, max_l=5, solvable_prob=0.5):
        self.min_n = min_n
        self.max_n = max_n
        self.min_s = min_s
        self.max_s = max_s
        self.min_l = min_l
        self.max_l = max_l
        self.solvable_prob = solvable_prob

    def case_generator(self):
        while True:
            n = random.randint(self.min_n, self.max_n)
            if n == 0:
                continue
            is_solvable = random.random() < self.solvable_prob

            if is_solvable:
                l = random.randint(self.min_l, min(n, self.max_l))
                if n < l:
                    continue
                max_possible_k = n // l
                if max_possible_k == 0:
                    continue
                k = random.randint(1, max_possible_k)
                lengths = self._generate_segment_lengths(n, k, l)
                s = random.randint(self.min_s, self.max_s)
                current_start = 0
                a = []
                for length in lengths:
                    segment = [random.randint(current_start, current_start + s) for _ in range(length)]
                    a.extend(segment)
                    current_start += s + 1
                expected = k
                break
            else:
                if n < 2:
                    continue
                l = random.randint(max(2, self.min_l), min(n, self.max_l))
                s = random.randint(self.min_s, self.max_s)
                base = 0
                a = []
                for i in range(n):
                    if i % l == (l - 1):
                        a.append(base + s + 1)
                    else:
                        a.append(base)
                expected = -1
                break

        return {
            'n': n,
            's': s,
            'l': l,
            'a': a,
            'expected': expected
        }

    def _generate_segment_lengths(self, total, k, l):
        base_length = l
        lengths = [base_length] * k
        remaining = total - base_length * k
        for i in range(remaining):
            lengths[i % k] += 1
        return lengths

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        s = question_case['s']
        l_val = question_case['l']
        a_str = ' '.join(map(str, question_case['a']))
        return f"""Alexandra has a paper strip with {n} numbers: {a_str}.
Each piece must have at least {l_val} numbers, and the difference between the maximum and minimum number in each piece must not exceed {s}.
Find the minimal number of pieces needed. If impossible, output -1.

Put your final answer within [answer] and [/answer] tags. For example:
[answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == identity['expected']:
            return True

        n, s, l = identity['n'], identity['s'], identity['l']
        arr = identity['a']
        arr = [0] + arr
        n_ref = len(arr) - 1
        dp = [float('inf')] * (n_ref + 1)
        dp[0] = 0

        for end in range(1, n_ref + 1):
            max_q = deque()
            min_q = deque()
            start = 0
            for i in range(end, 0, -1):
                while max_q and arr[i] >= arr[max_q[-1]]:
                    max_q.pop()
                max_q.append(i)
                while min_q and arr[i] <= arr[min_q[-1]]:
                    min_q.pop()
                min_q.append(i)

                while max_q and min_q and arr[max_q[0]] - arr[min_q[0]] > s:
                    if max_q[0] > min_q[0]:
                        start = max_q.popleft()
                    else:
                        start = min_q.popleft()

                if end - i + 1 >= l and i - 1 <= end - l:
                    if dp[i-1] + 1 < dp[end]:
                        dp[end] = dp[i-1] + 1

        valid_solution = dp[n_ref] if dp[n_ref] != float('inf') else -1
        return solution == valid_solution
