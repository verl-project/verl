"""# 

### 谜题描述
Mahmoud has an array a consisting of n integers. He asked Ehab to find another array b of the same length such that:

  * b is lexicographically greater than or equal to a. 
  * bi ≥ 2. 
  * b is pairwise coprime: for every 1 ≤ i < j ≤ n, bi and bj are coprime, i. e. GCD(bi, bj) = 1, where GCD(w, z) is the greatest common divisor of w and z. 



Ehab wants to choose a special array so he wants the lexicographically minimal array between all the variants. Can you find it?

An array x is lexicographically greater than an array y if there exists an index i such than xi > yi and xj = yj for all 1 ≤ j < i. An array x is equal to an array y if xi = yi for all 1 ≤ i ≤ n.

Input

The first line contains an integer n (1 ≤ n ≤ 105), the number of elements in a and b.

The second line contains n integers a1, a2, ..., an (2 ≤ ai ≤ 105), the elements of a.

Output

Output n space-separated integers, the i-th of them representing bi.

Examples

Input

5
2 3 5 4 13


Output

2 3 5 7 11 

Input

3
10 3 7


Output

10 3 7 

Note

Note that in the second sample, the array is already pairwise coprime so we printed it.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
MAX_NUM = 2000000
prime_str = ('2      3      5      7     11     13     17     19     23     29 '
        +   '31     37     41     43     47     53     59     61     67     71 '
        +   '73     79     83     89     97    101    103    107    109    113 '
        +  '127    131    137    139    149    151    157    163    167    173 '
        +  '179    181    191    193    197    199    211    223    227    229 '
        +  '233    239    241    251    257    263    269    271    277    281 '
        +  '283    293    307    311    313    317 '
)
prime_list = [int(p) for p in prime_str.split()]
used = [False] * MAX_NUM
n = input()
a = list(map(int,raw_input().split()))
def record(x):
    t = []
    for p in prime_list:
        if x % p == 0:
            while x % p == 0:
                x = x // p
            t.append(p)
            if x == 1:
                break
    if x != 1:
        t.append(x)

    for ti in t:
        for i in range(ti, MAX_NUM, ti):
            used[i] = True
b = []
for ai in a:
    if not used[ai]:
        b.append(ai)
        record(ai)
    else:
        temp = ai + 1
        while used[temp]:
            temp += 1
        b.append(temp)
        record(temp)
        break
temp = 2
while len(b) < len(a):
    while used[temp]:
        temp += 1
    b.append(temp)
    record(temp)
print(' '.join(str(x) for x in b))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Dmahmoudandehabandanotherarrayconstructiontaskbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_ai=100):
        self.max_n = max_n
        self.max_ai = max_ai

    @staticmethod
    def generate_b(a):
        MAX_NUM = 2000000
        prime_str = ('2 3 5 7 11 13 17 19 23 29 '
                     '31 37 41 43 47 53 59 61 67 71 '
                     '73 79 83 89 97 101 103 107 109 113 '
                     '127 131 137 139 149 151 157 163 167 173 '
                     '179 181 191 193 197 199 211 223 227 229 '
                     '233 239 241 251 257 263 269 271 277 281 '
                     '283 293 307 311 313 317')
        prime_list = [int(p) for p in prime_str.split()]
        used = [False] * (MAX_NUM + 1)
        n = len(a)
        b = []

        def record(x):
            t = []
            tmp_x = x
            for p in prime_list:
                if tmp_x % p == 0:
                    while tmp_x % p == 0:
                        tmp_x = tmp_x // p
                    t.append(p)
                    if tmp_x == 1:
                        break
            if tmp_x != 1:
                t.append(tmp_x)
            for ti in t:
                if ti > MAX_NUM:
                    continue
                for i in range(ti, MAX_NUM + 1, ti):
                    used[i] = True

        for ai in a:
            if ai <= MAX_NUM and not used[ai]:
                b.append(ai)
                record(ai)
            else:
                temp = ai + 1
                while temp <= MAX_NUM and used[temp]:
                    temp += 1
                if temp > MAX_NUM:
                    temp = ai + 1
                b.append(temp)
                record(temp)
                break  # Break after first replacement
        
        temp = 2
        while len(b) < len(a):
            while temp <= MAX_NUM and used[temp]:
                temp += 1
            if temp > MAX_NUM:
                break
            b.append(temp)
            record(temp)
            temp += 1
        
        return b

    def case_generator(self):
        n = random.randint(1, self.max_n)
        a = [random.randint(2, self.max_ai) for _ in range(n)]
        expected_b = self.generate_b(a)
        return {
            'n': n,
            'a': a,
            'expected_b': expected_b
        }

    @staticmethod
    def prompt_func(question_case):
        a_str = ' '.join(map(str, question_case['a']))
        return (
            f"You are given an array a of {question_case['n']} integers. Your task is to construct the lexicographically smallest array b that meets the following conditions:\n\n"
            "1. The array b is lexicographically greater than or equal to a.\n"
            "2. Each element in b is at least 2.\n"
            "3. All elements in b are pairwise coprime (their GCD must be 1).\n\n"
            f"Input array a: {a_str}\n\n"
            "Output the space-separated elements of array b. Enclose your answer within [answer] and [/answer] tags."
        )

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            return list(map(int, last_answer.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected_b = identity.get('expected_b', [])
        return solution == expected_b
