"""# 

### 谜题描述
There are n boys and m girls attending a theatre club. To set a play \"The Big Bang Theory\", they need to choose a group containing exactly t actors containing no less than 4 boys and no less than one girl. How many ways are there to choose a group? Of course, the variants that only differ in the composition of the troupe are considered different.

Perform all calculations in the 64-bit type: long long for С/С++, int64 for Delphi and long for Java.

Input

The only line of the input data contains three integers n, m, t (4 ≤ n ≤ 30, 1 ≤ m ≤ 30, 5 ≤ t ≤ n + m).

Output

Find the required number of ways.

Please do not use the %lld specificator to read or write 64-bit integers in С++. It is preferred to use cin, cout streams or the %I64d specificator.

Examples

Input

5 2 5


Output

10


Input

4 3 5


Output

3

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
fat = [0]*61
fat[0]=1
for i in xrange(1,61):
    fat[i] = fat[i-1]*i

n,m,t = map(int,raw_input().split())

resposta = fat[m+n]/(fat[t]*fat[m+n-t])
if n>=t:
    resposta-=fat[n]/(fat[t]*fat[n-t])
if m>=t:
    resposta-=fat[m]/(fat[t]*fat[m-t])
if m+3>=t:
    resposta-=fat[n]/(fat[3]*fat[n-3])*fat[m]/(fat[t-3]*fat[m-t+3])
if m+2>=t:
    resposta-=fat[n]/(fat[2]*fat[n-2])*fat[m]/(fat[t-2]*fat[m-t+2])
if m+1>=t:
    resposta-=fat[n]/(fat[1]*fat[n-1])*fat[m]/(fat[t-1]*fat[m-t+1])
print resposta
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

class Ctheworldisatheatrebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 4)
        self.n_max = params.get('n_max', 30)
        self.m_min = params.get('m_min', 1)
        self.m_max = params.get('m_max', 30)
    
    def case_generator(self):
        # 生成满足基本约束的案例（数学上保证t的有效性）
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        t_min = 5
        t_max = n + m
        t = random.randint(t_min, t_max)
        return {'n': n, 'm': m, 't': t}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        t = question_case['t']
        return f"""The theatre club has {n} boys and {m} girls. Select exactly {t} actors with:
- At least 4 boys
- At least 1 girl

Calculate the number of valid groups. Put the final answer within [answer][/answer] tags.

Examples:
Input: 5 2 5 → Output: 10
Input: 4 3 5 → Output: 3"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n, m, t = identity['n'], identity['m'], identity['t']
        
        total = math.comb(n + m, t)
        total -= math.comb(n, t)                     # 全男
        total -= math.comb(m, t)                     # 全女
        total -= math.comb(n,3) * math.comb(m,t-3)   # 3男
        total -= math.comb(n,2) * math.comb(m,t-2)   # 2男 
        total -= math.comb(n,1) * math.comb(m,t-1)   # 1男
        return solution == total
