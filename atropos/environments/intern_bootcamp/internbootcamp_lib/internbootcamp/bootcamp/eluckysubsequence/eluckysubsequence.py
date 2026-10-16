"""# 

### 谜题描述
Petya loves lucky numbers very much. Everybody knows that lucky numbers are positive integers whose decimal record contains only the lucky digits 4 and 7. For example, numbers 47, 744, 4 are lucky and 5, 17, 467 are not.

Petya has sequence a consisting of n integers.

The subsequence of the sequence a is such subsequence that can be obtained from a by removing zero or more of its elements.

Two sequences are considered different if index sets of numbers included in them are different. That is, the values ​of the elements ​do not matter in the comparison of subsequences. In particular, any sequence of length n has exactly 2n different subsequences (including an empty subsequence).

A subsequence is considered lucky if it has a length exactly k and does not contain two identical lucky numbers (unlucky numbers can be repeated any number of times).

Help Petya find the number of different lucky subsequences of the sequence a. As Petya's parents don't let him play with large numbers, you should print the result modulo prime number 1000000007 (109 + 7).

Input

The first line contains two integers n and k (1 ≤ k ≤ n ≤ 105). The next line contains n integers ai (1 ≤ ai ≤ 109) — the sequence a. 

Output

On the single line print the single number — the answer to the problem modulo prime number 1000000007 (109 + 7).

Examples

Input

3 2
10 10 10


Output

3


Input

4 2
4 4 7 7


Output

4

Note

In the first sample all 3 subsequences of the needed length are considered lucky.

In the second sample there are 4 lucky subsequences. For them the sets of indexes equal (the indexation starts from 1): {1, 3}, {1, 4}, {2, 3} and {2, 4}.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
MOD = 1000000007

n, m = map(int, raw_input().split())
arr = map(int, raw_input().split())

def check(num):
    return set(str(num)) <= set(['4', '7'])

ct = {}
nonlucky = 0
for num in arr:
    if not check(num):
        nonlucky += 1
        continue

    if num not in ct:
        ct[num] = 0

    ct[num] += 1


N = len(ct)
dp = [0] * (N+1)
dp[0] = 1
for num, amount in ct.items():
    for i in xrange(N, 0, -1):
        dp[i] = (dp[i] + dp[i-1] * amount)%MOD

mul = [0] * (n+1)
inv = [0] * (n+1)

mul[0] = 1
for i in xrange(1, n+1):
    mul[i] = (mul[i-1] * i)%MOD

def invP(num, P):
    if num == 1:
        return 1
    return ( invP(P%num, P)*(P - P/num) )%P

inv[n] = invP( mul[n], MOD )

for i in xrange(n-1, -1, -1):
    inv[i] = (inv[i+1] * (i+1))%MOD

def C(n, m):
    if n < 0 or m < 0 or n < m:
        return 0
    return (mul[n] * (inv[m] * inv[n-m])%MOD )%MOD

ans = 0
for i in xrange(N+1):
    ans = (ans + C(nonlucky, m - i) * dp[i]) % MOD

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

def is_lucky(num):
    s = str(num)
    for c in s:
        if c not in {'4', '7'}:
            return False
    return True

def generate_lucky_number():
    length = random.randint(1, 9)
    s = ''.join(random.choices(['4', '7'], k=length))
    return int(s)

def compute_answer(n, k, a):
    ct = {}
    nonlucky = 0
    for num in a:
        if is_lucky(num):
            if num not in ct:
                ct[num] = 0
            ct[num] += 1
        else:
            nonlucky += 1
    N = len(ct)
    dp = [0] * (N + 1)
    dp[0] = 1
    for num, amount in ct.items():
        for i in range(N, 0, -1):
            dp[i] = (dp[i] + dp[i-1] * amount) % MOD
    max_n = max(n, nonlucky)
    mul = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        mul[i] = mul[i-1] * i % MOD
    inv = [1] * (max_n + 1)
    inv[max_n] = pow(mul[max_n], MOD - 2, MOD)
    for i in range(max_n - 1, -1, -1):
        inv[i] = inv[i + 1] * (i + 1) % MOD
    def C(n, m):
        if n < 0 or m < 0 or n < m:
            return 0
        return mul[n] * inv[m] % MOD * inv[n - m] % MOD
    ans = 0
    for i in range(0, N + 1):
        if i > k:
            continue
        c = C(nonlucky, k - i)
        ans = (ans + dp[i] * c) % MOD
    return ans

class Eluckysubsequencebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', random.randint(1, 1000))
        self.k = params.get('k', random.randint(1, self.n))
        self.a = []
        self.correct_answer = 0
    
    def case_generator(self):
        n = self.n
        k = self.k
        a = []
        # 确保至少有k个幸运数字
        lucky_count = random.randint(k, n)
        nonlucky_count = n - lucky_count
        # 生成幸运数字
        for _ in range(lucky_count):
            a.append(generate_lucky_number())
        # 生成非幸运数字
        for _ in range(nonlucky_count):
            num = random.randint(1, 1000000)
            while is_lucky(num):
                num = random.randint(1, 1000000)
            a.append(num)
        # 计算正确答案
        correct_answer = compute_answer(n, k, a)
        self.a = a
        self.correct_answer = correct_answer
        return {
            'n': n,
            'k': k,
            'a': a,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        n = question_case['n']
        k = question_case['k']
        a_str = ', '.join(map(str, a))
        prompt = f"""
        Petya loves lucky numbers, which are positive integers consisting only of the digits 4 and 7. 
        Given a sequence of integers {a}, find the number of different lucky subsequences of length exactly {k}. 
        A lucky subsequence must satisfy the following conditions:
        1. The length of the subsequence is exactly {k}.
        2. The subsequence does not contain two identical lucky numbers.
        Non-lucky numbers can repeat any number of times.
        Your answer should be the number of such subsequences modulo 1000000007, placed within [answer] and [/answer] tags.
        """
        return prompt.strip()
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        if matches:
            return int(matches[-1])
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_answer = identity['correct_answer']
        if solution is None:
            return False
        return solution % MOD == correct_answer
