"""# 

### 谜题描述
Yaroslav thinks that two strings s and w, consisting of digits and having length n are non-comparable if there are two numbers, i and j (1 ≤ i, j ≤ n), such that si > wi and sj < wj. Here sign si represents the i-th digit of string s, similarly, wj represents the j-th digit of string w.

A string's template is a string that consists of digits and question marks (\"?\").

Yaroslav has two string templates, each of them has length n. Yaroslav wants to count the number of ways to replace all question marks by some integers in both templates, so as to make the resulting strings incomparable. Note that the obtained strings can contain leading zeroes and that distinct question marks can be replaced by distinct or the same integers.

Help Yaroslav, calculate the remainder after dividing the described number of ways by 1000000007 (109 + 7).

Input

The first line contains integer n (1 ≤ n ≤ 105) — the length of both templates. The second line contains the first template — a string that consists of digits and characters \"?\". The string's length equals n. The third line contains the second template in the same format.

Output

In a single line print the remainder after dividing the answer to the problem by number 1000000007 (109 + 7).

Examples

Input

2
90
09


Output

1


Input

2
11
55


Output

0


Input

5
?????
?????


Output

993531194

Note

The first test contains no question marks and both strings are incomparable, so the answer is 1.

The second test has no question marks, but the given strings are comparable, so the answer is 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import *
from itertools   import *
mod = 10**9+7
mul = lambda x,y: (x*y)%mod

def count_leq(ss,tt):
  leq = {'??': 55}
  for i in xrange(10):
    leq['?%d'%i] = i+1
    leq['%d?'%i] = 10-i
  for i in xrange(10):
    for j in xrange(10):
      leq['%d%d'%(i,j)] = 1 if i<=j else 0

  return reduce(mul, (leq[c1+c2] for c1,c2 in izip(ss,tt)), 1)

def count_eq(ss,tt):
  eq = defaultdict(int)
  for i in xrange(10):
    eq['%d%d'%(i,i)] = 1
    eq['?%d'%i] = 1
    eq['%d?'%i] = 1
  eq['??'] = 10
  return reduce(mul, (eq[c1+c2] for c1,c2 in izip(ss,tt)), 1)

n = int(raw_input())
s = raw_input()
t = raw_input()
total = pow(10, s.count('?')+t.count('?'),mod)
total -= count_leq(s,t)
total -= count_leq(t,s)
total += count_eq (t,s)
print total%mod
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from functools import reduce
from collections import defaultdict
from bootcamp import Basebootcamp

class Byaroslavandtwostringsbootcamp(Basebootcamp):
    mod = 10**9 + 7

    def __init__(self, n=None, min_n=1, max_n=5, p_question=0.3):
        self.n = n  # Fixed length mode
        self.min_n = min(n, min_n) if n else min_n  # 确保范围有效性
        self.max_n = max(n, max_n) if n else max_n
        self.p_question = min(max(p_question, 0.0), 1.0)  # 概率范围约束

    def case_generator(self):
        n = self.n if self.n is not None else random.randint(self.min_n, self.max_n)
        return {
            'n': n,
            's': self._generate_template(n),
            't': self._generate_template(n)
        }

    def _generate_template(self, n):
        return ''.join(
            '?' if random.random() < self.p_question else str(random.randint(0, 9)) 
            for _ in range(n)
        )

    @staticmethod
    def prompt_func(question_case):
        return f"""给定两个长度为{question_case['n']}的字符串模板：
模板1：{question_case['s']}
模板2：{question_case['t']}

计算所有问号替换为数字的方案中，使得这两个字符串满足以下条件的方案数（模1e9+7）：
- 存在至少一个位置i使得模板1的第i位数字 > 模板2的第i位数字
- 同时存在至少一个位置j使得模板1的第j位数字 < 模板2的第j位数字

答案请置于[answer]标签内，示例：[answer]123456789[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) % Byaroslavandtwostringsbootcamp.mod if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 优化大数运算的分步计算
        s, t = identity['s'], identity['t']
        mod = cls.mod
        
        # 计算总可能性
        total = pow(10, s.count('?') + t.count('?'), mod)
        
        # 并行计算三种量
        leq_st = 1
        leq_ts = 1
        eq = 1
        
        pre_cache = defaultdict(lambda: defaultdict(int))
        # 预计算所有字符组合
        for a in '0123456789?':
            for b in '0123456789?':
                leq = 0
                geq = 0
                equal = 0
                for x in (range(10) if a == '?' else [int(a)]):
                    for y in (range(10) if b == '?' else [int(b)]):
                        leq += (x <= y)
                        geq += (x >= y)
                        equal += (x == y)
                pre_cache['leq'][a+b] = leq % mod
                pre_cache['geq'][a+b] = geq % mod
                pre_cache['eq'][a+b] = equal % mod
        
        for c1, c2 in zip(s, t):
            key = c1 + c2
            leq_st = (leq_st * pre_cache['leq'][key]) % mod
            leq_ts = (leq_ts * pre_cache['geq'][key]) % mod
            eq = (eq * pre_cache['eq'][key]) % mod
        
        correct = (total - leq_st - leq_ts + eq) % mod
        return solution == correct
