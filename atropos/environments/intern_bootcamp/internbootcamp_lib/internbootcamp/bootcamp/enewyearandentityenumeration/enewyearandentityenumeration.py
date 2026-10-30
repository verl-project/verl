"""# 

### 谜题描述
You are given an integer m.

Let M = 2m - 1.

You are also given a set of n integers denoted as the set T. The integers will be provided in base 2 as n binary strings of length m.

A set of integers S is called \"good\" if the following hold. 

  1. If <image>, then <image>. 
  2. If <image>, then <image>
  3. <image>
  4. All elements of S are less than or equal to M. 



Here, <image> and <image> refer to the bitwise XOR and bitwise AND operators, respectively.

Count the number of good sets S, modulo 109 + 7.

Input

The first line will contain two integers m and n (1 ≤ m ≤ 1 000, 1 ≤ n ≤ min(2m, 50)).

The next n lines will contain the elements of T. Each line will contain exactly m zeros and ones. Elements of T will be distinct.

Output

Print a single integer, the number of good sets modulo 109 + 7. 

Examples

Input

5 3
11010
00101
11000


Output

4


Input

30 2
010101010101010010101010101010
110110110110110011011011011011


Output

860616440

Note

An example of a valid set S is {00000, 00101, 00010, 00111, 11000, 11010, 11101, 11111}.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#Nice now I'm sumproduct of gps times bell function
mod=10**9+7
def Blist(m):
    A = [0 for i in range(0, m)]
    A[0] = 1
    R = [1, 1]
    for n in range(1, m):
        A[n] = A[0]
        for k in range(n, 0, -1):
            A[k-1] += A[k]
            A[k-1]%=mod
        R.append(A[0])
    return R
    
    
m,n=[int(k) for k in raw_input().split(\" \")]
t=[]
for i in range(n):
    t+=[[k for k in raw_input()]]

ti=[int(\"\".join([t[i][k] for i in range(n)]),2) for k in range(m)]

left=set(range(m))
potes=[]
gps=[]
mxl=0
for k in range(m):
    if k in left:
        totej=set()
        for j in left:
            if ti[k]^ti[j]==0:
                totej.add(j)
        left=left-totej
        gps+=[len(totej)]  
        mxl=max(mxl,len(totej))

bl=Blist(m)

res=1

for k in gps:
    res*=bl[k]
    res%=mod

print res
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

mod = 10**9 + 7

class Enewyearandentityenumerationbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.m_min = params.get('m_min', 1)
        self.m_max = params.get('m_max', 5)
    
    @staticmethod
    def _generate_binary_strings(m, n):
        binaries = set()
        while len(binaries) < n:
            num = random.randint(0, (1 << m) - 1)
            binary = bin(num)[2:].zfill(m)
            binaries.add(binary)
        return list(binaries)
    
    @staticmethod
    def _Blist(m_val):
        A = [0] * m_val
        A[0] = 1
        R = [1, 1]
        for n in range(1, m_val):
            A[n] = A[0]
            for k in range(n, 0, -1):
                A[k-1] += A[k]
                A[k-1] %= mod
            R.append(A[0])
        return R
    
    @staticmethod
    def _compute_answer(m, T):
        n = len(T)
        t = [list(s) for s in T]
        ti = [int(''.join(row[k] for row in t), 2) for k in range(m)]
        left = set(range(m))
        gps = []
        while left:
            k = next(iter(left))
            current = ti[k]
            group = {j for j in left if ti[j] == current}
            left -= group
            gps.append(len(group))
        bell_numbers = Enewyearandentityenumerationbootcamp._Blist(m)
        res = 1
        for size in gps:
            res = res * bell_numbers[size] % mod
        return res
    
    def case_generator(self):
        m = random.randint(self.m_min, self.m_max)
        max_n = min(2**m, 50)
        n = random.randint(1, max_n)
        T = self._generate_binary_strings(m, n)
        correct_answer = self._compute_answer(m, T)
        return {
            'm': m,
            'n': n,
            'T': T,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        m = question_case['m']
        n = question_case['n']
        T = question_case['T']
        T_str = '\n'.join(T)
        return f"""You are given an integer m = {m} and a set T of {n} distinct binary strings of length {m}. Determine the number of good sets S modulo 10^9 + 7.

A good set S must satisfy:
1. For any x, y in S, x ^ y is in S.
2. For any x, y in S, x & y is in S.
3. All elements of T are in S.
4. Every element in S ≤ 2^{m} - 1.

Input Format:
{m} {n}
{T_str}

Output Format:
A single integer, the count modulo 10^9 + 7.

Example:
Input:
5 3
11010
00101
11000
Output:
4

Place your answer within [answer] and [/answer] tags, e.g., [answer]4[/answer]."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer', None)
