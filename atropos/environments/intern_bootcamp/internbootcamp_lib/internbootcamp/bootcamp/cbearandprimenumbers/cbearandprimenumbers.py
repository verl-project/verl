"""# 

### 谜题描述
Recently, the bear started studying data structures and faced the following problem.

You are given a sequence of integers x1, x2, ..., xn of length n and m queries, each of them is characterized by two integers li, ri. Let's introduce f(p) to represent the number of such indexes k, that xk is divisible by p. The answer to the query li, ri is the sum: <image>, where S(li, ri) is a set of prime numbers from segment [li, ri] (both borders are included in the segment).

Help the bear cope with the problem.

Input

The first line contains integer n (1 ≤ n ≤ 106). The second line contains n integers x1, x2, ..., xn (2 ≤ xi ≤ 107). The numbers are not necessarily distinct.

The third line contains integer m (1 ≤ m ≤ 50000). Each of the following m lines contains a pair of space-separated integers, li and ri (2 ≤ li ≤ ri ≤ 2·109) — the numbers that characterize the current query.

Output

Print m integers — the answers to the queries on the order the queries appear in the input.

Examples

Input

6
5 5 7 10 14 15
3
2 11
3 12
4 4


Output

9
7
0


Input

7
2 3 5 7 11 4 8
2
8 10
2 123


Output

0
7

Note

Consider the first sample. Overall, the first sample has 3 queries.

  1. The first query l = 2, r = 11 comes. You need to count f(2) + f(3) + f(5) + f(7) + f(11) = 2 + 1 + 4 + 2 + 0 = 9. 
  2. The second query comes l = 3, r = 12. You need to count f(3) + f(5) + f(7) + f(11) = 1 + 4 + 2 + 0 = 7. 
  3. The third query comes l = 4, r = 4. As this interval has no prime numbers, then the sum equals 0. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin,stdout,setrecursionlimit,maxint,exit
setrecursionlimit(2*10**5)
#from os.path import dirname
#currdir=dirname(__file__)
#stdin=open(currdir+\"/inputfile\",\"r\")
#stdout=open(currdir+\"/output1.txt\",\"w\")
def listInput():
 return map(long,stdin.readline().split())
def printBS(li):
 if not li: return
 for i in xrange(len(li)-1):
  stdout.write(\"%d \"%li[i])
 stdout.write(\"%d\n\"%li[-1])
def sin():
 return stdin.readline().rstrip()
n=int(sin())
li=listInput()
N=max(li)
done=[0]*(N+1)
nos=[0]*(N+1)
primCount=[0]*(N+1)
for i in li:
 nos[i]+=1
for i in xrange(2,N+1):
  if done[i]==0:
   for j in xrange(i,N+1,i):
    done[j]=1
    if nos[j]: primCount[i]+=nos[j]
for i in xrange(1,N+1):
 primCount[i]+=primCount[i-1]
m=int(sin())
#print primCount
for i in xrange(m):
 l,r=listInput()
 if r>N: r=N
 if l>N: l=N+1
 stdout.write(\"%d\n\"%(primCount[r]-primCount[l-1]))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from math import isqrt
from bootcamp import Basebootcamp

class Cbearandprimenumbersbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_x=10**4, max_m=5):
        self.max_n = max_n
        self.max_x = max_x
        self.max_m = max_m
    
    def case_generator(self):
        # 生成保证至少有一个素数查询的测试案例
        n = random.randint(1, self.max_n)
        x = []
        while True:
            x = [random.randint(2, self.max_x) for _ in range(n)]
            if any(num >= 2 for num in x):  # 确保至少有一个可能的素数
                break
        
        # 预处理素数频次
        max_x_val = max(x) if x else 0
        sieve_max = max_x_val
        
        # 使用正确的筛法统计素数频次
        prime_counts = defaultdict(int)
        num_counts = defaultdict(int)
        for num in x:
            num_counts[num] += 1
        
        # 计算每个数的素数因子
        for num, cnt in num_counts.items():
            temp = num
            for p in self.get_primes(num):
                if temp % p == 0:
                    prime_counts[p] += cnt
                    while temp % p == 0:
                        temp //= p
            if temp > 1:  # 剩余的大素数
                prime_counts[temp] += cnt
        
        # 生成素数列表并排序
        primes = sorted(prime_counts.keys())
        if not primes:
            primes = [2]  # 防止空列表情况
        
        # 构建前缀和数组
        max_prime = primes[-1] if primes else 0
        prefix = [0] * (max_prime + 2)
        for p in primes:
            if p <= max_prime:
                prefix[p] = prime_counts[p]
        
        # 计算前缀和
        for i in range(1, len(prefix)):
            prefix[i] += prefix[i-1]
        
        # 生成查询
        m = random.randint(1, self.max_m)
        queries = []
        answers = []
        valid_queries = 0
        
        while valid_queries < m:
            # 生成包含有效素数的查询
            if random.random() < 0.5 and primes:
                # 生成覆盖素数的查询
                p = random.choice(primes)
                l = random.randint(max(2, p-2), p)
                r = random.randint(p, min(2*10**9, p+2))
            else:
                l = random.randint(2, 2*10**9)
                r = random.randint(l, 2*10**9)
            
            # 计算正确答案
            r_clamped = min(r, max_prime)
            if l > r_clamped:
                ans = 0
            else:
                ans = prefix[r_clamped] - prefix[l-1]
            
            # 确保至少有一定比例的查询有非零答案
            if valid_queries < m//2 and ans == 0:
                continue
            
            queries.append((l, r))
            answers.append(ans)
            valid_queries += 1

        return {
            'n': n,
            'x': x,
            'm': m,
            'queries': queries,
            'answers': answers,
            '_prefix': prefix,
            '_max_prime': max_prime
        }

    @staticmethod
    def get_primes(n):
        """返回n的所有素数因子"""
        primes = set()
        if n < 2:
            return primes
        while n % 2 == 0:
            primes.add(2)
            n //= 2
        i = 3
        max_factor = isqrt(n) + 1
        while i <= max_factor and n > 1:
            while n % i == 0:
                primes.add(i)
                n //= i
                max_factor = isqrt(n) + 1
            i += 2
        if n > 1:
            primes.add(n)
        return primes
    
    @staticmethod
    def prompt_func(question_case):
        problem = (
            "You are given a sequence of integers and need to process multiple queries. Each query asks for the sum of f(p) for all primes p in a given range [l, r]. "
            "Here, f(p) is the number of integers in the sequence divisible by p.\n\n"
            "Input format:\n"
            "- First line: integer n (1 ≤ n ≤ 1e6)\n"
            "- Second line: n integers x1, x2, ..., xn (2 ≤ xi ≤ 1e7)\n"
            "- Third line: integer m (1 ≤ m ≤ 50000)\n"
            "- Next m lines: pairs of integers li, ri (2 ≤ li ≤ ri ≤ 2e9)\n\n"
            f"Current input:\n{question_case['n']}\n{' '.join(map(str, question_case['x']))}\n"
            f"{question_case['m']}\n" + 
            '\n'.join(f"{l} {r}" for l, r in question_case['queries']) + 
            "\n\nOutput m integers in order, each on a new line. Enclose your answer with [answer] tags.\n"
            "Example format:\n[answer]\n3\n0\n5\n[/answer]"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        numbers = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                numbers.extend(re.findall(r'-?\d+', line))
        try:
            return list(map(int, numbers))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answers', [])
