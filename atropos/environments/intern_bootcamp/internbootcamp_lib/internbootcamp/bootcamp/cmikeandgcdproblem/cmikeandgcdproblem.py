"""# 

### 谜题描述
Mike has a sequence A = [a1, a2, ..., an] of length n. He considers the sequence B = [b1, b2, ..., bn] beautiful if the gcd of all its elements is bigger than 1, i.e. <image>. 

Mike wants to change his sequence in order to make it beautiful. In one move he can choose an index i (1 ≤ i < n), delete numbers ai, ai + 1 and put numbers ai - ai + 1, ai + ai + 1 in their place instead, in this order. He wants perform as few operations as possible. Find the minimal number of operations to make sequence A beautiful if it's possible, or tell him that it is impossible to do so.

<image> is the biggest non-negative number d such that d divides bi for every i (1 ≤ i ≤ n).

Input

The first line contains a single integer n (2 ≤ n ≤ 100 000) — length of sequence A.

The second line contains n space-separated integers a1, a2, ..., an (1 ≤ ai ≤ 109) — elements of sequence A.

Output

Output on the first line \"YES\" (without quotes) if it is possible to make sequence A beautiful by performing operations described above, and \"NO\" (without quotes) otherwise.

If the answer was \"YES\", output the minimal number of moves needed to make sequence A beautiful.

Examples

Input

2
1 1


Output

YES
1


Input

3
6 2 4


Output

YES
0


Input

2
1 3


Output

YES
1

Note

In the first example you can simply make one move to obtain sequence [0, 2] with <image>.

In the second example the gcd of the sequence is already greater than 1. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from fractions import gcd

n = int(raw_input())
p = [int(x) for x in raw_input().split()]

g = p[0]
ans = 0
cnt = 0

for x in p:
    g = gcd(x,g)
    if(x %2 == 0):
        ans += (cnt/2) + 2*(cnt%2)
        cnt = 0
    else: cnt += 1

ans += (cnt/2) + 2*(cnt%2)

print \"YES\"
if g > 1:
    print 0
else:
    print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from math import gcd
from functools import reduce
from bootcamp import Basebootcamp

class Cmikeandgcdproblembootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10**5, max_value=10**9):
        self.n_min = n_min
        self.n_max = n_max
        self.max_value = max_value
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        # 决定生成类型：50%概率生成可立即解决的案例，50%需要计算的案例
        if random.choice([True, False]):
            # 生成初始gcd>1的情况
            d = random.randint(2, 5)
            a = [d * random.randint(1, self.max_value//d) for _ in range(n)]
        else:
            # 生成初始gcd=1的数组（必须通过操作解决）
            while True:
                # 生成至少两个互质数确保整体gcd=1
                a = []
                # 生成前两个互质数
                x, y = random.sample(range(1, self.max_value//2 +1), 2)
                while gcd(x, y) != 1:
                    x, y = random.sample(range(1, self.max_value//2 +1), 2)
                a.extend([x, y])
                # 生成其余元素
                for _ in range(n-2):
                    a.append(random.randint(1, self.max_value))
                # 验证整体gcd
                if reduce(gcd, a) == 1:
                    break
        return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        a_str = ' '.join(map(str, question_case['a']))
        return f"""给定长度为{question_case['n']}的数列: {a_str}
请你判断是否可以通过替换相邻两数为它们的差和和（操作次数最少），使得数列所有元素的最大公约数大于1。若可以，输出YES并给出最少操作数，否则输出NO。答案格式：[answer]YES\nX[/answer] 或 [answer]NO[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
        content = matches[-1].strip().upper()
        lines = [line.strip() for line in content.split('\n')]
        if not lines:
            return None
        if lines[0] == 'YES':
            if len(lines) < 2:
                return None
            try:
                steps = int(lines[1])
                return ('YES', steps)
            except ValueError:
                return None
        elif lines[0] == 'NO':
            return ('NO', None)
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        current_gcd = reduce(gcd, a)
        
        # 初始gcd>1的验证
        if current_gcd > 1:
            return solution == ('YES', 0)
        
        # 必须通过操作的情况
        cnt = ans = 0
        for x in a:
            if x % 2 == 0:
                ans += (cnt // 2) + 2 * (cnt % 2)
                cnt = 0
            else:
                cnt += 1
        ans += (cnt // 2) + 2 * (cnt % 2)
        return solution == ('YES', ans)
