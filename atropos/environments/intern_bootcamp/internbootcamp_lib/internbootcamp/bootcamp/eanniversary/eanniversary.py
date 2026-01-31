"""# 

### 谜题描述
There are less than 60 years left till the 900-th birthday anniversary of a famous Italian mathematician Leonardo Fibonacci. Of course, such important anniversary needs much preparations.

Dima is sure that it'll be great to learn to solve the following problem by the Big Day: You're given a set A, consisting of numbers l, l + 1, l + 2, ..., r; let's consider all its k-element subsets; for each such subset let's find the largest common divisor of Fibonacci numbers with indexes, determined by the subset elements. Among all found common divisors, Dima is interested in the largest one.

Dima asked to remind you that Fibonacci numbers are elements of a numeric sequence, where F1 = 1, F2 = 1, Fn = Fn - 1 + Fn - 2 for n ≥ 3.

Dima has more than half a century ahead to solve the given task, but you only have two hours. Count the residue from dividing the sought largest common divisor by m.

Input

The first line contains four space-separated integers m, l, r and k (1 ≤ m ≤ 109; 1 ≤ l < r ≤ 1012; 2 ≤ k ≤ r - l + 1).

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use cin, cout streams or the %I64d specifier.

Output

Print a single integer — the residue from dividing the sought greatest common divisor by m.

Examples

Input

10 1 8 2


Output

3


Input

10 1 8 3


Output

1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#coding=utf-8


def recfib(n,m):
    if n==0: return (0,1,)
    a, b = recfib(n / 2,m)
    return ((b*b+a*a)%m, b*(2*a+b)%m) if n%2 else (a*((2*b)-a)%m, ((b*b+a*a))%m)


m,l,r,k=map(int,raw_input().split())

#print recfib(small,m)[0]
#print recfib(big,m)[0]
n = (r-l)/(k-1)
while n >= 0 and r/n - (l-1)/n < k:
   # print n
    n = r/(r/n+1)
    
print recfib(n,m)[0]
#print recfib(3362719727,m)[0]
#for i in range(12):
#	print \"Fib(%d)=%d\"%(i,recfib(i,10000)[0])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Eanniversarybootcamp(Basebootcamp):
    def __init__(self):
        pass
    
    def case_generator(self):
        m = random.randint(1, 10**9)
        l = random.randint(1, 10**12)
        r = random.randint(l + 1, 10**12)
        k = random.randint(2, r - l + 1)
        return {
            'm': m,
            'l': l,
            'r': r,
            'k': k
        }
    
    @staticmethod
    def prompt_func(question_case):
        m = question_case['m']
        l = question_case['l']
        r = question_case['r']
        k = question_case['k']
        prompt = (
            f"给定四个整数 m = {m}, l = {l}, r = {r}, k = {k}。\n"
            "你需要解决的问题是：找到区间 [l, r] 中所有 k 个元素的子集，"
            "计算每个子集对应斐波那契数索引的最大公约数，然后找出这些公约数中的最大值。"
            "将这个最大公约数代入斐波那契数列中，计算其模 m 的值。\n"
            "斐波那契数列定义为：F1 = 1, F2 = 1, Fn = Fn-1 + Fn-2，n ≥ 3。\n"
            "请将答案以整数形式放在 [answer] 标签内。例如，如果计算结果是 3，"
            "那么回答应为：[answer]3[/answer]。"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if not last_match.isdigit():
            return None
        return int(last_match)
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        m = identity['m']
        l = identity['l']
        r = identity['r']
        k = identity['k']
        
        def find_max_d(l, r, k):
            low = 1
            high = r
            max_d = 0
            while low <= high:
                mid = (low + high) // 2
                count = (r // mid) - ((l - 1) // mid)
                if count >= k:
                    max_d = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return max_d
        
        d = find_max_d(l, r, k)
        
        def fib(n, mod):
            a, b = 0, 1
            for _ in range(n):
                a, b = b, (a + b) % mod
            return a
        
        if d == 0:
            correct = 0
        else:
            # 计算 F(d) mod m
            # 由于 d 可能很大，使用迭代方法避免栈溢出
            correct = fib(d, m)
        
        return solution == correct
