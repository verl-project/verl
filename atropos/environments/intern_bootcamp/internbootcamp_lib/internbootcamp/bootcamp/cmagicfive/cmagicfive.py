"""# 

### 谜题描述
There is a long plate s containing n digits. Iahub wants to delete some digits (possibly none, but he is not allowed to delete all the digits) to form his \"magic number\" on the plate, a number that is divisible by 5. Note that, the resulting number may contain leading zeros.

Now Iahub wants to count the number of ways he can obtain magic number, modulo 1000000007 (109 + 7). Two ways are different, if the set of deleted positions in s differs.

Look at the input part of the statement, s is given in a special form.

Input

In the first line you're given a string a (1 ≤ |a| ≤ 105), containing digits only. In the second line you're given an integer k (1 ≤ k ≤ 109). The plate s is formed by concatenating k copies of a together. That is n = |a|·k.

Output

Print a single integer — the required number of ways modulo 1000000007 (109 + 7).

Examples

Input

1256
1


Output

4


Input

13990
2


Output

528


Input

555
2


Output

63

Note

In the first case, there are four possible ways to make a number that is divisible by 5: 5, 15, 25 and 125.

In the second case, remember to concatenate the copies of a. The actual plate is 1399013990.

In the third case, except deleting all digits, any choice will do. Therefore there are 26 - 1 = 63 possible ways to delete digits.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#! /usr/bin/python

__author__ = 'Muntasir Khan'

import sys

MOD = 1000000007


def mod_exp(base, exp, mod):
  if exp == 0:
    return 1
  elif exp&1:
    return (base * mod_exp(base, exp - 1, mod))
  else:
    x = mod_exp(base, exp >> 1, mod)
    return (x * x) % mod


def ways(s, k):
  k_left = k - 1
  ret = 0
  two_l = (1<<len(s)) % MOD
  denom = mod_exp((two_l - 1 + MOD) % MOD, MOD - 2, MOD)
  power_sum = mod_exp(two_l, k, MOD)
  
  base_sum = 0
  pwr = 1
  
  for index, char in enumerate(s):
    if char in ('0', '5'):
      base_sum = (base_sum + pwr) % MOD
    pwr = (pwr << 1) % MOD 
  
  return (base_sum * (power_sum - 1 +MOD) * denom) % MOD


def main():
  s = sys.stdin.readline().strip()
  k = int(sys.stdin.readline())
  if s:
    print ways(s, k) 
  else:
    print '0'


if __name__ == '__main__':
  main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Cmagicfivebootcamp(Basebootcamp):
    def __init__(self, a_min_length=1, a_max_length=5, k_min=1, k_max=1000, prob_05=0.4):
        self.a_min_length = max(1, a_min_length)
        self.a_max_length = max(a_min_length, a_max_length)
        self.k_min = max(1, k_min)
        self.k_max = max(k_min, k_max)
        self.prob_05 = prob_05

    def case_generator(self):
        if random.random() < self.prob_05:
            a = self._gen_has_05()
        else:
            a = self._gen_random()
        k = random.randint(self.k_min, self.k_max)
        return {'a': a, 'k': k}

    def _gen_has_05(self):
        length = random.randint(self.a_min_length, self.a_max_length)
        chars = []
        for _ in range(length):
            if random.random() < 0.3:
                chars.append(random.choice(['0', '5']))
            else:
                chars.append(random.choice('12346789'))
        if not any(c in {'0','5'} for c in chars):
            chars[random.randint(0, len(chars)-1)] = random.choice(['0','5'])
        return ''.join(chars)

    def _gen_random(self):
        length = random.randint(self.a_min_length, self.a_max_length)
        return ''.join(random.choices('0123456789', k=length))

    @staticmethod
    def prompt_func(question_case) -> str:
        a = question_case['a']
        k = question_case['k']
        return f"""You are trying to solve a mathematical puzzle involving divisibility by 5. The puzzle's details are as follows:

Problem Description:
You are given a string 'a' consisting of digits and an integer 'k'. The string 's' is formed by concatenating 'a' exactly 'k' times. For example, if a is "12" and k is 3, then s is "121212".

Your task is to determine the number of distinct ways to delete some (but not all) digits from 's' such that the remaining digits form a number divisible by 5. The result should be computed modulo 1e9+7 (1000000007).

Rules:
1. A valid way of deletion must leave at least one digit remaining.
2. Two deletion methods are considered different if the sets of positions deleted are different, even if the resulting number is the same.
3. Leading zeros in the resulting number are allowed. For example, "005" is a valid number if divisible by 5.

Input Details:
- The string 'a' is "{a}" (length: {len(a)} characters).
- The integer 'k' is {k}.

Your task is to compute the number of valid deletion ways, taking into account the modulo requirement. Provide your final answer as an integer inside [answer] tags. For example, if your answer is 42, write [answer]42[/answer]."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        k = identity['k']
        try:
            correct = cls.compute_ways(a, k)
            return solution == correct
        except:
            return False

    @staticmethod
    def compute_ways(a, k):
        MOD = 10**9 + 7
        if not a:
            return 0
            
        l = len(a)
        two_l = pow(2, l, MOD)
        denominator = (two_l - 1) % MOD
        inv_denominator = pow(denominator, MOD-2, MOD) if denominator != 0 else 0
        
        power_sum = pow(two_l, k, MOD)
        
        base_sum = 0
        pwr = 1  # 对应2^0
        for char in a:
            if char in {'0', '5'}:
                base_sum = (base_sum + pwr) % MOD
            pwr = (pwr * 2) % MOD
        
        if inv_denominator == 0:
            total = 0
        else:
            numerator = (base_sum * (power_sum - 1 + MOD)) % MOD
            total = (numerator * inv_denominator) % MOD
        return total
