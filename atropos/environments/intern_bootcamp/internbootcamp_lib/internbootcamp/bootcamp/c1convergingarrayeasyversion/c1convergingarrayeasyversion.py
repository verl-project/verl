"""# 

### 谜题描述
This is the easy version of the problem. The only difference is that in this version q = 1. You can make hacks only if both versions of the problem are solved.

There is a process that takes place on arrays a and b of length n and length n-1 respectively. 

The process is an infinite sequence of operations. Each operation is as follows: 

  * First, choose a random integer i (1 ≤ i ≤ n-1). 
  * Then, simultaneously set a_i = min\left(a_i, \frac{a_i+a_{i+1}-b_i}{2}\right) and a_{i+1} = max\left(a_{i+1}, \frac{a_i+a_{i+1}+b_i}{2}\right) without any rounding (so values may become non-integer). 

See notes for an example of an operation.

It can be proven that array a converges, i. e. for each i there exists a limit a_i converges to. Let function F(a, b) return the value a_1 converges to after a process on a and b.

You are given array b, but not array a. However, you are given a third array c. Array a is good if it contains only integers and satisfies 0 ≤ a_i ≤ c_i for 1 ≤ i ≤ n.

Your task is to count the number of good arrays a where F(a, b) ≥ x for q values of x. Since the number of arrays can be very large, print it modulo 10^9+7.

Input

The first line contains a single integer n (2 ≤ n ≤ 100).

The second line contains n integers c_1, c_2 …, c_n (0 ≤ c_i ≤ 100).

The third line contains n-1 integers b_1, b_2, …, b_{n-1} (0 ≤ b_i ≤ 100).

The fourth line contains a single integer q (q=1).

The fifth line contains q space separated integers x_1, x_2, …, x_q (-10^5 ≤ x_i ≤ 10^5).

Output

Output q integers, where the i-th integer is the answer to the i-th query, i. e. the number of good arrays a where F(a, b) ≥ x_i modulo 10^9+7.

Example

Input


3
2 3 4
2 1
1
-1


Output


56

Note

The following explanation assumes b = [2, 1] and c=[2, 3, 4] (as in the sample).

Examples of arrays a that are not good: 

  * a = [3, 2, 3] is not good because a_1 > c_1; 
  * a = [0, -1, 3] is not good because a_2 < 0. 



One possible good array a is [0, 2, 4]. We can show that no operation has any effect on this array, so F(a, b) = a_1 = 0.

Another possible good array a is [0, 1, 4]. In a single operation with i = 1, we set a_1 = min((0+1-2)/(2), 0) and a_2 = max((0+1+2)/(2), 1). So, after a single operation with i = 1, a becomes equal to [-1/2, 3/2, 4]. We can show that no operation has any effect on this array, so F(a, b) = -1/2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division,print_function
from heapq import*
import sys
le = sys.__stdin__.read().split(\"\n\")[::-1]
mo=10**9+7
if 1:
    n=int(le.pop())
    c = list(map(int,le.pop().split()))
    b = list(map(int,le.pop().split()))
    le.pop()
    x=int(le.pop())
    prefb=[b[0]]
    for k in b[1:]:
        prefb.append(k+prefb[-1])
    prefb.append(0)    #careful when going to C
    prefbt=[0]
    for k in range(1,n-1):
        prefbt.append(k*b[k]+prefbt[-1])
    prefbt.append(0)
sc=sum(c)
d=[[0]*(sc+1) for k in range(n+1)]#fuck a lot of prefix sum in this problem
ds=[[0]*(sc+2) for k in range(n+1)]#prefix array beginning with 0
ds[-1]=list(range(sc+2))
d[-1]=[1]*(sc+1)
for index in range(n-1,-1,-1):
    #can even go linearly for minpref
    minpref=0
    while (minpref-index*prefb[index-1]+prefbt[index-1])/(index+1)<x:
       minpref+=1
    for pref in range(sc+1):
        mi=min(pref+c[index]+1,sc+1)
        ma=max(minpref,pref)
        d[index][pref]=0 if mi<ma else ds[index+1][mi]-ds[index+1][ma]
    for pref in range(1,sc+2):
        ds[index][pref]=(ds[index][pref-1]+d[index][pref-1])%mo
print(d[0][0]%mo)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random

MOD = 10**9 + 7

class C1convergingarrayeasyversionbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=5, c_max=100, x_min=-100, x_max=100):
        self.n_min = n_min
        self.n_max = n_max
        self.c_max = c_max
        self.x_min = x_min
        self.x_max = x_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        c = [random.randint(0, self.c_max) for _ in range(n)]
        b = [random.randint(0, self.c_max) for _ in range(n-1)]
        x = random.randint(self.x_min, self.x_max)
        return {
            'n': n,
            'c': c,
            'b': b,
            'x_list': [x]
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        c = question_case['c']
        b = question_case['b']
        x = question_case['x_list'][0]
        problem = (
            "You are tasked with counting the number of valid arrays 'a' under specific constraints.\n\n"
            f"- Array 'a' has length {n} with each element a_i being an integer where 0 ≤ a_i ≤ {c}.\n"
            f"- Array 'b' is given as {b}.\n"
            "An infinite process modifies 'a' through random operations. Each operation selects an index i and updates:\n"
            "  a_i = min(a_i, (a_i + a_{i+1} - b_i)/2)\n"
            "  a_{i+1} = max(a_{i+1}, (a_i + a_{i+1} + b_i)/2)\n"
            "The function F(a, b) returns the limit of a₁ after infinite operations.\n\n"
            f"**Task**: Count how many valid arrays 'a' satisfy F(a, b) ≥ {x}. Return the result modulo 10⁹+7.\n"
            "Format your answer as an integer within [answer]...[/answer]."
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        numbers = re.findall(r'-?\d+', last_match)
        if not numbers:
            return None
        try:
            return int(numbers[-1]) % MOD
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            c = identity['c']
            b = identity['b']
            x = identity['x_list'][0]
            correct = cls.compute_answer(n, c, b, x)
            return solution == correct
        except:
            return False
    
    @staticmethod
    def compute_answer(n, c, b, x_val):
        prefb = []
        if b:
            prefb = [b[0]]
            for k in b[1:]:
                prefb.append(k + prefb[-1])
        prefb.append(0)
        prefbt = [0]
        for k in range(1, n-1):
            if k < len(b):
                term = k * b[k] + prefbt[-1]
                prefbt.append(term)
        sc = sum(c)
        d = [[0]*(sc+1) for _ in range(n+1)]
        ds = [[0]*(sc+2) for _ in range(n+1)]
        ds[-1] = list(range(sc+2))
        d[-1] = [1]*(sc+1)
        for index in range(n-1, -1, -1):
            minpref = 0
            if index >= 1:
                denominator = index + 1
                numerator = minpref - index * prefb[index-1] + (prefbt[index-1] if (index-1) < len(prefbt) else 0)
                while denominator != 0 and numerator / denominator < x_val:
                    minpref += 1
                    numerator = minpref - index * prefb[index-1] + (prefbt[index-1] if (index-1) < len(prefbt) else 0)
            else:
                denominator = index + 1
                numerator = minpref - 0 * prefb[-1] + (prefbt[-1] if prefbt else 0)
                while denominator != 0 and numerator / denominator < x_val:
                    minpref += 1
                    numerator = minpref - 0 * prefb[-1] + (prefbt[-1] if prefbt else 0)
            for pref in range(sc+1):
                mi = min(pref + c[index] + 1, sc+1)
                ma = max(minpref, pref)
                if mi < ma:
                    d[index][pref] = 0
                else:
                    d[index][pref] = (ds[index+1][mi] - ds[index+1][ma]) % MOD
            for pref in range(1, sc+2):
                ds[index][pref] = (ds[index][pref-1] + d[index][pref-1]) % MOD
        return d[0][0] % MOD
