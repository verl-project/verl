"""# 

### 谜题描述
As Famil Door’s birthday is coming, some of his friends (like Gabi) decided to buy a present for him. His friends are going to buy a string consisted of round brackets since Famil Door loves string of brackets of length n more than any other strings!

The sequence of round brackets is called valid if and only if: 

  1. the total number of opening brackets is equal to the total number of closing brackets; 
  2. for any prefix of the sequence, the number of opening brackets is greater or equal than the number of closing brackets. 



Gabi bought a string s of length m (m ≤ n) and want to complete it to obtain a valid sequence of brackets of length n. He is going to pick some strings p and q consisting of round brackets and merge them in a string p + s + q, that is add the string p at the beginning of the string s and string q at the end of the string s.

Now he wonders, how many pairs of strings p and q exists, such that the string p + s + q is a valid sequence of round brackets. As this number may be pretty large, he wants to calculate it modulo 109 + 7.

Input

First line contains n and m (1 ≤ m ≤ n ≤ 100 000, n - m ≤ 2000) — the desired length of the string and the length of the string bought by Gabi, respectively.

The second line contains string s of length m consisting of characters '(' and ')' only.

Output

Print the number of pairs of string p and q such that p + s + q is a valid sequence of round brackets modulo 109 + 7.

Examples

Input

4 1
(


Output

4


Input

4 4
(())


Output

1


Input

4 3
(((


Output

0

Note

In the first sample there are four different valid pairs: 

  1. p = \"(\", q = \"))\" 
  2. p = \"()\", q = \")\" 
  3. p = \"\", q = \"())\" 
  4. p = \"\", q = \")()\" 



In the second sample the only way to obtain a desired string is choose empty p and q.

In the third sample there is no way to get a valid sequence of brackets.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def main():
    n, m = map(int, raw_input().split())
    s = raw_input().strip()
    c = d = 0
    for x in s:
        c += (x == '(') * 2 - 1
        d = min(c, d)
    d = -d
    mod = 1000000007
    dp = [[1]]
    p = dp[-1]
    for i in xrange(n - m):
        ndp = p[1:] + [0] * 2
        for j in xrange(1, i + 2):
            ndp[j] += p[j-1]
            if ndp[j] >= mod:
                ndp[j] -= mod
        dp.append(ndp)
        p = ndp
    ans = 0
    for i in xrange(n - m + 1):
        l = n - m - i
        p = dp[l]
        for j in xrange(d, min(l + 1 - c, i + 1)):
            ans += dp[i][j] * p[j+c]
            ans %= mod
    print ans
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cfamildoorandbracketsbootcamp(Basebootcamp):
    def __init__(self, max_n=100000, max_diff=2000, min_m=1, mod=10**9+7):
        self.max_n = max_n
        self.max_diff = max_diff
        self.min_m = min_m
        self.mod = mod
    
    def case_generator(self):
        max_k = self.max_diff
        # Ensure n <= max_n and n-m <= max_diff
        m = random.randint(self.min_m, self.max_n)
        k = random.randint(0, min(max_k, self.max_n - m))
        n = m + k
        # Generate non-empty s of length m
        s = ''.join(random.choice(['(', ')']) for _ in range(m))
        return {'n': n, 'm': m, 's': s}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        s = question_case['s']
        return f"""Given a bracket string "{s}" of length {m}, find the number of valid bracket sequences of length {n} formed by adding prefixes and suffixes. Return mod 1e9+7. Put the answer within [answer] tags."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            m = identity['m']
            s = identity['s']
            return solution == cls.calculate(n, m, s)
        except:
            return False

    @staticmethod
    def calculate(n, m, s):
        MOD = 10**9+7
        balance = min_balance = 0
        for c in s:
            balance += 1 if c == '(' else -1
            min_balance = min(min_balance, balance)
        
        req = -min_balance
        total = n - m
        if balance > total or (balance + total) % 2 != 0:
            return 0

        max_len = total
        # DP[i][j] = number of valid sequences of length i with balance j
        dp = [[0]*(max_len+2) for _ in range(max_len+1)]
        dp[0][0] = 1
        for i in range(max_len):
            for j in range(max_len+1):
                if not dp[i][j]:
                    continue
                # Add '('
                if j+1 <= max_len:
                    dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % MOD
                # Add ')' only if balance stays non-negative
                if j-1 >= 0:
                    dp[i+1][j-1] = (dp[i+1][j-1] + dp[i][j]) % MOD

        ans = 0
        for p_len in range(total+1):
            q_len = total - p_len
            if q_len < 0:
                continue
            for p_bal in range(req, p_len+1):
                if (p_bal + p_len) % 2:  # Balance parity check
                    continue
                # q must have balance = - (p_bal + balance_s)
                needed_q_bal = -(p_bal + balance)
                needed_q_len = q_len
                if needed_q_bal < 0 or needed_q_bal > needed_q_len:
                    continue
                if (needed_q_len - needed_q_bal) % 2:
                    continue
                ans = (ans + dp[p_len][p_bal] * dp[q_len][needed_q_bal]) % MOD
        return ans
