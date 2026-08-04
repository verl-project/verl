"""# 

### 谜题描述
Kaavi, the mysterious fortune teller, deeply believes that one's fate is inevitable and unavoidable. Of course, she makes her living by predicting others' future. While doing divination, Kaavi believes that magic spells can provide great power for her to see the future. 

<image>

Kaavi has a string T of length m and all the strings with the prefix T are magic spells. Kaavi also has a string S of length n and an empty string A.

During the divination, Kaavi needs to perform a sequence of operations. There are two different operations:

  * Delete the first character of S and add it at the front of A.
  * Delete the first character of S and add it at the back of A.



Kaavi can perform no more than n operations. To finish the divination, she wants to know the number of different operation sequences to make A a magic spell (i.e. with the prefix T). As her assistant, can you help her? The answer might be huge, so Kaavi only needs to know the answer modulo 998 244 353.

Two operation sequences are considered different if they are different in length or there exists an i that their i-th operation is different. 

A substring is a contiguous sequence of characters within a string. A prefix of a string S is a substring of S that occurs at the beginning of S.

Input

The first line contains a string S of length n (1 ≤ n ≤ 3000).

The second line contains a string T of length m (1 ≤ m ≤ n).

Both strings contain only lowercase Latin letters.

Output

The output contains only one integer — the answer modulo 998 244 353.

Examples

Input


abab
ba


Output


12

Input


defineintlonglong
signedmain


Output


0

Input


rotator
rotator


Output


4

Input


cacdcdbbbb
bdcaccdbbb


Output


24

Note

The first test:

<image>

The red ones are the magic spells. In the first operation, Kaavi can either add the first character \"a\" at the front or the back of A, although the results are the same, they are considered as different operations. So the answer is 6×2=12.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

MOD = 998244353
def modder(x):
    return x if x < MOD else x - MOD

S = input()
T = input()

n = len(S)
m = len(T)

DP = [[0]*n for _ in range(n + 1)]

c = S[0]
for i in range(n):
    DP[1][i] = 2 * (i >= m or c == T[i])

for l in range(1, n):
    DPl = DP[l]
    DPlp1 = DP[l + 1]

    for i in range(n - l + 1):
        c = S[l]
        if i and (i - 1 >= m or T[i - 1] == c):
            DPlp1[i - 1] = modder(DPlp1[i - 1] + DPl[i])

        if i + l < n and (i + l >= m or T[i + l] == c):
            DPlp1[i] += modder(DPlp1[i] + DPl[i])

print sum(DP[j][0] for j in range(m, n + 1)) % MOD
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Ckaaviandmagicspellbootcamp(Basebootcamp):
    def __init__(self, max_n=3000, min_m=1, default_m=2, default_n=4):
        self.params = {
            'max_n': max_n,
            'min_m': min_m,
            'default_m': default_m,
            'default_n': default_n
        }
    
    def case_generator(self):
        m = random.randint(self.params['min_m'], self.params['default_m'])
        n = random.randint(m, self.params['default_n'])
        
        # 生成保证至少存在一个解的测试用例
        T = ''.join(random.choices('abc', k=m))
        
        # 构造合法S的核心部分（必须与T前缀匹配）
        op_sequence = []
        constructed = list(T)
        for _ in range(m):
            op = random.choice(['front', 'back'])
            op_sequence.append(op)
            if op == 'front':
                constructed.pop(0)  # 逆向构造
            else:
                constructed.pop()
        
        S_core = ''.join(constructed)
        
        # 补充随机字符
        if n > m:
            S_core += ''.join(random.choices('abc', k=n - m))
        
        return {"S": S_core, "T": T}

    @staticmethod
    def prompt_func(question_case):  # 原第44行
        S = question_case['S']
        T = question_case['T']
        prompt = f"""Ckaaviandmagicspell needs to determine the number of valid operation sequences when building string A from "{S}" that results in the prefix "{T}". 
Each operation chooses to prepend/append the next character from S. 
The answer must be a single integer within [answer][/answer] tags."""
        return prompt

    @staticmethod
    def extract_output(output):  # 此处修复缩进问题
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        MOD = 998244353
        S = identity['S']
        T = identity['T']
        n, m = len(S), len(T)
        
        # 边界条件处理
        if m == 0 or n < m:
            return solution == 0
        
        # 动态规划验证核心逻辑
        dp = [[0]*(n+1) for _ in range(n+1)]
        dp[0][0] = 1
        
        for step in range(n):
            for pos in range(n+1):
                if dp[step][pos] == 0:
                    continue
                
                c = S[step]
                # 前置分支
                if pos > 0 and (pos-1 >= m or T[pos-1] == c):
                    dp[step+1][pos-1] = (dp[step+1][pos-1] + dp[step][pos]) % MOD
                # 后置分支
                end_pos = pos + step
                if end_pos >= m or (end_pos < m and T[end_pos] == c):
                    dp[step+1][pos] = (dp[step+1][pos] + dp[step][pos]) % MOD
        
        total = sum(dp[j][0] for j in range(m, n+1)) % MOD
        return solution == total
