"""# 

### 谜题描述
Vasya has recently bought some land and decided to surround it with a wooden fence.

He went to a company called \"Wooden board\" that produces wooden boards for fences. Vasya read in the catalog of products that the company has at its disposal n different types of wood. The company uses the i-th type of wood to produce a board of this type that is a rectangular ai by bi block.

Vasya decided to order boards in this company and build a fence from them. It turned out that the storehouse of the company is so large that Vasya can order arbitrary number of boards of every type. Note that Vasya is allowed to turn the boards as he builds the fence. However, Vasya cannot turn square boards.

Vasya is required to construct a fence of length l, however, an arbitrary fence won't do. Vasya wants his fence to look beautiful. We'll say that a fence is beautiful if and only if the following two conditions are fulfilled:

  * there are no two successive boards of the same type 
  * the first board of the fence has an arbitrary length, and the length of each subsequent board equals the width of the previous one 



In other words, the fence is considered beautiful, if the type of the i-th board in the fence is different from the i - 1-th board's type; besides, the i-th board's length is equal to the i - 1-th board's width (for all i, starting from 2).

Now Vasya wonders, how many variants of arranging a fence for his land exist. Your task is to count the number of different beautiful fences of length l.

Two fences will be considered the same if the corresponding sequences of fence boards types and rotations are the same, otherwise the fences are different. Since the sought number can be large enough, you need to calculate the answer modulo 1000000007 (109 + 7).

Input

The first line contains two integers n and l (1 ≤ n ≤ 100, 1 ≤ l ≤ 3000) — the number of different board types and the fence length, correspondingly. Next n lines contain descriptions of board types: the i-th line contains two integers ai and bi (1 ≤ ai, bi ≤ 100) — the sizes of the board of the i-th type. All numbers on the lines are separated by spaces.

Output

Print a single integer — the sought number of variants modulo 1000000007 (109 + 7).

Examples

Input

2 3
1 2
2 3


Output

2


Input

1 2
2 2


Output

1


Input

6 6
2 1
3 2
2 5
3 3
5 1
2 1


Output

20

Note

In the first sample there are exactly two variants of arranging a beautiful fence of length 3: 

  * As the first fence board use the board of the first type of length 1 and width 2. As the second board use board of the second type of length 2 and width 3. 
  * Use one board of the second type after you turn it. That makes its length equal 3, and width — 2. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin

m = 1000000007
n, l = map(int, stdin.readline().split())
e = []
for i in xrange(n):
    a, b = map(int, stdin.readline().split())
    e.append((a, b, i))
    if a != b: e.append((b, a, i))
n = len(e)
e.sort()

u = [[] for i in xrange(101)]
for i in xrange(n):
    u[e[i][1]].append((e[i][0], e[i][1], e[i][2], i))

d = [[0] * n for i in xrange(l + 1)]
for i in xrange(l + 1):
    for j in reversed(xrange(i)):
        r = i - j
        if r > 100: break
        f = 0
        while f < n:
            if e[f][0] > r: break
            if e[f][0] == r:
                if j == 0:
                    d[i][f] = 1
                else:
                    g = 0
                    while g < len(u[r]):
                        if e[f][2] != u[r][g][2]:
                            d[i][f] = (d[i][f] + d[j][u[r][g][3]]) % m
                        g += 1
            f += 1

print sum(d[-1]) % m
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Ewoodenfencebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'min_n': 1,
            'max_n': 100,
            'min_l': 1,
            'max_l': 3000,
            'min_size': 1,
            'max_size': 100
        }
        self.params.update(params)
    
    def case_generator(self):
        n = random.randint(self.params['min_n'], self.params['max_n'])
        l_val = random.randint(self.params['min_l'], self.params['max_l'])
        types = []
        for _ in range(n):
            a = random.randint(self.params['min_size'], self.params['max_size'])
            b = random.randint(self.params['min_size'], self.params['max_size'])
            types.append([a, b])
        return {
            'n': n,
            'l': l_val,
            'types': types
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        l_val = question_case['l']
        types = question_case['types']
        types_str = '\n'.join([f"{a} {b}" for a, b in types])
        prompt = f"""Vasya has bought a piece of land and wants to surround it with a beautiful wooden fence using boards from the "Wooden board" company. The company provides {n} different types of boards. Each board type is a rectangular piece with specific dimensions. Vasya can order any number of boards of each type and can rotate them unless they are square. The beautiful fence must satisfy two conditions:

1. No two consecutive boards are of the same type.
2. The length of each subsequent board must equal the width of the previous one.

The total length of the fence must be exactly {l_val}. Your task is to calculate the number of different valid beautiful fences Vasya can build, modulo 1,000,000,007.

Input format:
- The first line contains two integers, n (1 ≤ n ≤ 100) and l (1 ≤ l ≤ 3000).
- The next n lines each contain two integers a_i and b_i (1 ≤ a_i, b_i ≤ 100), describing the dimensions of each board type.

Problem data:
{n} {l_val}
{types_str}

Please compute the answer and place your final numerical answer within [answer] and [/answer] tags. For example, if your answer is 5, write [answer]5[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 直接使用参考算法验证答案
        MOD = 10**9 +7
        n_orig = identity['n']
        l_val = identity['l']
        types = identity['types']
        
        # 生成所有可能的板子方向
        expanded = []
        for idx, (a, b) in enumerate(types):
            expanded.append((a, b, idx))
            if a != b:
                expanded.append((b, a, idx))
        expanded.sort()
        
        # 构建状态转移表
        transition_table = [[] for _ in range(101)]
        for e_idx, (a, b, t_idx) in enumerate(expanded):
            transition_table[b].append((a, b, t_idx, e_idx))
        
        # 动态规划表
        dp = [[0]*len(expanded) for _ in range(l_val+1)]
        
        for curr_total in range(l_val+1):
            for prev_total in reversed(range(curr_total)):
                seg_len = curr_total - prev_total
                if seg_len > 100:
                    break
                
                ptr = 0
                while ptr < len(expanded):
                    a, b, t_idx = expanded[ptr]
                    if a > seg_len:
                        break
                    if a == seg_len:
                        if prev_total == 0:
                            dp[curr_total][ptr] = 1
                        else:
                            for prev_a, prev_b, prev_t, prev_e in transition_table[seg_len]:
                                if prev_t != t_idx:
                                    dp[curr_total][ptr] = (dp[curr_total][ptr] + dp[prev_total][prev_e]) % MOD
                    ptr += 1
        
        correct = sum(dp[l_val]) % MOD
        return solution == correct
