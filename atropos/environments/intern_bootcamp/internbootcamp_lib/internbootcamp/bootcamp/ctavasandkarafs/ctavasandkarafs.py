"""# 

### 谜题描述
Karafs is some kind of vegetable in shape of an 1 × h rectangle. Tavaspolis people love Karafs and they use Karafs in almost any kind of food. Tavas, himself, is crazy about Karafs.

<image>

Each Karafs has a positive integer height. Tavas has an infinite 1-based sequence of Karafses. The height of the i-th Karafs is si = A + (i - 1) × B.

For a given m, let's define an m-bite operation as decreasing the height of at most m distinct not eaten Karafses by 1. Karafs is considered as eaten when its height becomes zero.

Now SaDDas asks you n queries. In each query he gives you numbers l, t and m and you should find the largest number r such that l ≤ r and sequence sl, sl + 1, ..., sr can be eaten by performing m-bite no more than t times or print -1 if there is no such number r.

Input

The first line of input contains three integers A, B and n (1 ≤ A, B ≤ 106, 1 ≤ n ≤ 105).

Next n lines contain information about queries. i-th line contains integers l, t, m (1 ≤ l, t, m ≤ 106) for i-th query.

Output

For each query, print its answer in a single line.

Examples

Input

2 1 4
1 5 3
3 3 10
7 10 2
6 4 8


Output

4
-1
8
-1


Input

1 5 2
1 5 10
2 7 4


Output

1
2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
R=lambda:map(int,raw_input().split())
a,b,n=R()
for _ in range(n):
  l,t,m=R()
  lo,hi=0,10001000
  v=a+(l-1)*b
  while lo<hi:
    x=(lo+hi+1)/2
    if (v+v+(x-1)*b)*x>t*min(m,x)*2:
      hi=x-1
    else:
      lo=x
  print min(l+lo-1,(t-a)/b+1) if lo>0 else -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_r(A, B, l, t, m):
    v = A + (l - 1) * B
    if v > t:
        return -1

    lo, hi = 0, 10**8
    while lo < hi:
        mid = (lo + hi + 1) // 2
        sum_condition = (v + v + (mid - 1) * B) * mid
        right = t * min(m, mid) * 2
        if sum_condition > right:
            hi = mid - 1
        else:
            lo = mid

    if lo == 0:
        return -1

    max_r1 = l + lo - 1
    max_r2 = (t - A) // B + 1 if B != 0 else t
    r = min(max_r1, max_r2)
    return r if r >= l else -1

class Ctavasandkarafsbootcamp(Basebootcamp):
    def __init__(self, A=None, B=None, **params):
        super().__init__(**params)
        # 调整参数范围，生成更合理的测试案例
        self.A = A if A is not None else random.randint(1, 100)
        self.B = B if B is not None else random.randint(1, 100)

    def case_generator(self):
        # 控制参数范围以提高案例质量
        for _ in range(100):  # 防止无限循环
            l = random.randint(1, 50)
            s_l = self.A + (l-1)*self.B
            
            # 有50%概率生成有解案例
            if random.random() < 0.5:
                t = random.randint(s_l, s_l + 1000)
            else:
                t = random.randint(1, max(1, s_l - 1))
            
            m = random.randint(1, 100)
            expected_r = compute_r(self.A, self.B, l, t, m)
            
            # 确保生成的案例格式正确
            if expected_r != -1 or random.random() < 0.3:  # 保留部分无解案例
                return {
                    'A': self.A,
                    'B': self.B,
                    'l': l,
                    't': t,
                    'm': m,
                    'expected_r': expected_r
                }
        
        # 保底返回一个无解案例
        l = random.randint(1, 50)
        return {
            'A': self.A,
            'B': self.B,
            'l': l,
            't': random.randint(1, 10),
            'm': random.randint(1, 10),
            'expected_r': -1
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        A = question_case['A']
        B = question_case['B']
        l = question_case['l']
        t = question_case['t']
        m = question_case['m']
        prompt = f"""你正在帮助SaDDas解决关于Ctavasandkarafs的查询问题。Ctavasandkarafs按照无限序列排列，第i个的高度为s_i = {A} + (i-1) × {B}。每次操作（m-bite）可以选择最多m个不同的未被吃掉的Ctavasandkarafs，每个减少1点高度。当一个Ctavasandkarafs的高度变为0时被吃掉，无法再被选择。

给定查询参数：起始位置l={l}，最多允许t={t}次操作，每次操作最多选m={m}个Ctavasandkarafs。请找出最大的r满足以下条件：

1. l ≤ r；
2. 通过最多t次m-bite操作可以吃完第l到第r的所有Ctavasandkarafs。

如果不存在这样的r，请输出-1。答案必须是整数，格式为[answer]答案[/answer]，例如：[answer]5[/answer]或[answer]-1[/answer]。

当前问题参数：
A = {A}
B = {B}
l = {l}
t = {t}
m = {m}

注意：
1. 最终答案必须满足：max(s_l,...,s_r) ≤ t 且总操作次数足够
2. 答案只能放在[answer]标签内，其他位置将无法识别"""
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
        return solution == identity['expected_r']
