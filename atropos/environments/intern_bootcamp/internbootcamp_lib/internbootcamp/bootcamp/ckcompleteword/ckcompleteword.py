"""# 

### 谜题描述
Word s of length n is called k-complete if 

  * s is a palindrome, i.e. s_i=s_{n+1-i} for all 1 ≤ i ≤ n; 
  * s has a period of k, i.e. s_i=s_{k+i} for all 1 ≤ i ≤ n-k. 



For example, \"abaaba\" is a 3-complete word, while \"abccba\" is not.

Bob is given a word s of length n consisting of only lowercase Latin letters and an integer k, such that n is divisible by k. He wants to convert s to any k-complete word.

To do this Bob can choose some i (1 ≤ i ≤ n) and replace the letter at position i with some other lowercase Latin letter.

So now Bob wants to know the minimum number of letters he has to replace to convert s to any k-complete word.

Note that Bob can do zero changes if the word s is already k-complete.

You are required to answer t test cases independently.

Input

The first line contains a single integer t (1 ≤ t≤ 10^5) — the number of test cases.

The first line of each test case contains two integers n and k (1 ≤ k < n ≤ 2 ⋅ 10^5, n is divisible by k).

The second line of each test case contains a word s of length n.

It is guaranteed that word s only contains lowercase Latin letters. And it is guaranteed that the sum of n over all test cases will not exceed 2 ⋅ 10^5.

Output

For each test case, output one integer, representing the minimum number of characters he has to replace to convert s to any k-complete word.

Example

Input


4
6 2
abaaba
6 3
abaaba
36 9
hippopotomonstrosesquippedaliophobia
21 7
wudixiaoxingxingheclp


Output


2
0
23
16

Note

In the first test case, one optimal solution is aaaaaa.

In the second test case, the given word itself is k-complete.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
'''input
4
6 2
abaaba
6 3
abaaba
36 9
hippopotomonstrosesquippedaliophobia
21 7
wudixiaoxingxingheclp
'''

import sys
import math
from collections import Counter

debug = 1
readln = sys.stdin.readline
#sys.setrecursionlimit(1000000)

def write(s):
    sys.stdout.write(str(s))

def writeln(s):
    sys.stdout.write(str(s))
    sys.stdout.write('\n')

def readint():
    return int(readln().strip())

def readints():
    return map(int, readln().split())

def readstr():
    return readln().strip()

def readstrs():
    return readln().split()

def dbg(*args):
    if debug: print(' '.join(map(str, args)))

def solve(n,k,s):
    cts = [Counter() for _ in xrange(k)]
    for i,c in enumerate(s):
        b = 0
        if i % k < k/2:
            b = i % k
        else:
            b = k - i % k - 1
        cts[b][c] += 1
    ans = 0
    for ct in cts:
        if not ct: break
        common = ct.most_common(1)[0]
        ans += common[1]
    return n - ans

t = readint()
for _ in xrange(t):
    n,k = readints()
    s = readstr()
    writeln(solve(n,k,s))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
from collections import Counter
from bootcamp import Basebootcamp

class Ckcompletewordbootcamp(Basebootcamp):
    def __init__(self, max_k=5, max_m=5, **params):
        super().__init__(**params)
        self.max_k = max_k
        self.max_m = max_m

    def case_generator(self):
        while True:
            # Generate valid k and n with k < n
            k = random.randint(1, self.max_k)
            m = random.randint(2, self.max_m)
            n = k * m
            if k < n:
                break
        
        # Generate valid k-complete base structure
        group_size = (k + 1) // 2
        base_chars = [random.choice(string.ascii_lowercase) for _ in range(group_size)]
        
        # Build minimal k-complete string
        s = []
        for i in range(n):
            mod = i % k
            if mod >= group_size:
                mod = k - mod - 1  # Mirror for palindrome
            s.append(base_chars[mod % group_size])
        
        # Add controlled noise
        max_noise = min(n, max(1, n // 4))
        noise_count = random.randint(0, max_noise)
        noise_indices = random.sample(range(n), k=noise_count)
        for idx in noise_indices:
            available_chars = [c for c in string.ascii_lowercase if c != s[idx]]
            if available_chars:
                s[idx] = random.choice(available_chars)
        
        return {'n': n, 'k': k, 's': ''.join(s)}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        s = question_case['s']
        return f"""Given a word s of length {n} and integer k = {k}, find the minimum number of character changes needed to make it k-complete. The word is: "{s}"

A k-complete word must:
1. Be a palindrome
2. Have a period of k

Put your final answer between [answer] and [/answer] tags. Example: [answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == cls.solve(identity['n'], identity['k'], identity['s'])
        except:
            return False

    @staticmethod
    def solve(n, k, s):
        group_size = (k + 1) // 2
        groups = [Counter() for _ in range(group_size)]
        
        for i, c in enumerate(s):
            mod = i % k
            if mod >= group_size:
                mod = k - mod - 1
            groups[mod % group_size][c] += 1
        
        total = sum(max(ct.values(), default=0) for ct in groups)
        return n - total
