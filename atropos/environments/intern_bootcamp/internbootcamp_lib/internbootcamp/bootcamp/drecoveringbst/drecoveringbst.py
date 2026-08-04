"""# 

### 谜题描述
Dima the hamster enjoys nibbling different things: cages, sticks, bad problemsetters and even trees!

Recently he found a binary search tree and instinctively nibbled all of its edges, hence messing up the vertices. Dima knows that if Andrew, who has been thoroughly assembling the tree for a long time, comes home and sees his creation demolished, he'll get extremely upset. 

To not let that happen, Dima has to recover the binary search tree. Luckily, he noticed that any two vertices connected by a direct edge had their greatest common divisor value exceed 1.

Help Dima construct such a binary search tree or determine that it's impossible. The definition and properties of a binary search tree can be found [here.](https://en.wikipedia.org/wiki/Binary_search_tree)

Input

The first line contains the number of vertices n (2 ≤ n ≤ 700).

The second line features n distinct integers a_i (2 ≤ a_i ≤ 10^9) — the values of vertices in ascending order.

Output

If it is possible to reassemble the binary search tree, such that the greatest common divisor of any two vertices connected by the edge is greater than 1, print \"Yes\" (quotes for clarity).

Otherwise, print \"No\" (quotes for clarity).

Examples

Input

6
3 6 9 18 36 108


Output

Yes


Input

2
7 17


Output

No


Input

9
4 8 10 12 15 18 33 44 81


Output

Yes

Note

The picture below illustrates one of the possible trees for the first example.

<image>

The picture below illustrates one of the possible trees for the third example.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = sys.stdin.readline

n = int(input())
A = [int(x) for x in input().split()]

good_gcd = [False]*(710**2)
for i in range(n):
    for j in range(n):
        a,b = A[i],A[j]
        while b:
            a,b = b,a%b
        good_gcd[i+710*j]=a>1
    good_gcd[i+710*n]=True
    good_gcd[n+710*i]=True


mem = [-1]*(2*710**2)

# [a,b)
def can_bst(left,a,b):
    #assert(a<b)
    val = a-1 if left==0 else b
    
    if b-a==0:
        return True
    elif b-a==1:
        return good_gcd[a*710+val]
    
    
    key = a+b*710+left*(710**2)
    if mem[key]==-1:
        for root_a in range(a,b):
            if good_gcd[710*root_a+val] and can_bst(1,a,root_a) and can_bst(0,root_a+1,b):
                mem[key]=True
                break    
        else:
            mem[key]=False
    return mem[key]


if can_bst(1,0,n):
    print 'Yes'
else:
    print 'No'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import math
import random
from math import gcd
from collections import defaultdict

class Drecoveringbstbootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=15, yes_ratio=0.5, max_attempts=100):
        self.n_min = max(n_min, 2)
        self.n_max = min(n_max, 100)  # 控制最大规模
        self.yes_ratio = yes_ratio
        self.max_attempts = max_attempts
        self.prime_pool = self._sieve(200)  # 预备质数库

    def _sieve(self, n):
        sieve = [True] * (n+1)
        sieve[0:2] = [False]*2
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                sieve[i*i::i] = [False]*len(sieve[i*i::i])
        return [i for i, b in enumerate(sieve) if b]

    def case_generator(self):
        for _ in range(self.max_attempts):
            generate_yes = random.random() < self.yes_ratio
            
            if generate_yes:
                # Yes案例：构建保证有解的树结构
                case = self._generate_yes_case()
                if case:
                    return case
            else:
                # No案例：确保无解的构造
                case = self._generate_no_case()
                if case:
                    return case
        
        # 生成失败时返回标准案例
        return {
            'n': 2,
            'array': [2, 3],
            'expected_answer': 'No' if gcd(2,3)==1 else 'Yes'
        }

    def _generate_yes_case(self):
        """生成保证有解的案例：通过链式结构构造"""
        # 方法一：构建链式树（完全左/右子树）
        n = random.randint(self.n_min, self.n_max)
        base = random.choice([2, 3, 4, 5, 6])
        step = random.choice([2, 3, 4])
        arr = sorted([base * (step**i) for i in range(n)])
        
        # 方法二：共享因子的随机组合
        factors = random.sample(self.prime_pool, 3)
        candidates = []
        for _ in range(2*n):
            p = random.choice(factors)
            q = random.choice(factors)
            if p != q:
                candidates.append(p*q)
        arr = sorted(list(set(candidates)))[:n]
        if len(arr) < self.n_min:
            return None
        
        expected = self.check_possible(arr)
        if expected == 'Yes':
            return {
                'n': len(arr),
                'array': arr,
                'expected_answer': expected
            }
        return None

    def _generate_no_case(self):
        """生成保证无解的案例：互质数或特殊结构"""
        # 方法一：使用互质数
        primes = random.sample(self.prime_pool, self.n_max*2)
        arr = sorted(primes[:random.randint(self.n_min, self.n_max)])
        if all(math.gcd(a,b)==1 for a in arr for b in arr if a!=b):
            return {
                'n': len(arr),
                'array': arr,
                'expected_answer': 'No'
            }
        
        # 方法二：构造无法形成BST结构的案例
        while True:
            base = random.choice([2,3])
            arr = sorted([base**i for i in range(1, self.n_max+1)])
            if self.check_possible(arr) == 'No':
                return {
                    'n': len(arr),
                    'array': arr,
                    'expected_answer': 'No'
                }
            break
        
        return None

    @staticmethod
    def check_possible(a):
        # 优化后的验证算法（带记忆化）
        n = len(a)
        gcd_cache = [[math.gcd(a[i], a[j]) > 1 for j in range(n)] for i in range(n)]
        parent = [[-1]*n for _ in range(n)]
        dp = [[False]*n for _ in range(n)]

        # 构建根节点可能性
        for i in range(n):
            dp[i][i] = True

        # 区间DP
        for l in range(2, n+1):
            for i in range(n - l + 1):
                j = i + l - 1
                for k in range(i, j+1):
                    left_ok = (k == i) or (dp[i][k-1] and gcd_cache[k][k-1])
                    right_ok = (k == j) or (dp[k+1][j] and gcd_cache[k][k+1])
                    if left_ok and right_ok:
                        dp[i][j] = True
                        parent[i][j] = k
                        break

        return 'Yes' if dp[0][n-1] else 'No'

    @staticmethod
    def prompt_func(question_case):
        elements = ' '.join(map(str, question_case['array']))
        return f"""Determine if a valid BST can be built from these sorted values where adjacent nodes have GCD>1.

Input:
{question_case['n']}
{elements}

Output format: [answer]Yes/No[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](Yes|No)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1].capitalize() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_answer']
