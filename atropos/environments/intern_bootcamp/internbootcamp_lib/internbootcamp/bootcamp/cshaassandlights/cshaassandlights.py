"""# 

### 谜题描述
There are n lights aligned in a row. These lights are numbered 1 to n from left to right. Initially some of the lights are switched on. Shaass wants to switch all the lights on. At each step he can switch a light on (this light should be switched off at that moment) if there's at least one adjacent light which is already switched on. 

He knows the initial state of lights and he's wondering how many different ways there exist to switch all the lights on. Please find the required number of ways modulo 1000000007 (109 + 7).

Input

The first line of the input contains two integers n and m where n is the number of lights in the sequence and m is the number of lights which are initially switched on, (1 ≤ n ≤ 1000, 1 ≤ m ≤ n). The second line contains m distinct integers, each between 1 to n inclusive, denoting the indices of lights which are initially switched on.

Output

In the only line of the output print the number of different possible ways to switch on all the lights modulo 1000000007 (109 + 7).

Examples

Input

3 1
1


Output

1


Input

4 2
1 4


Output

2


Input

11 2
4 8


Output

6720

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from math import *
n,m=map(int,raw_input().split())
l=map(int,raw_input().split())
l.sort()
a=factorial(n-m)
i=0
f=0
while i<m:
    d=l[i]-f-1
    a/=factorial(d)
    if i>0:
        a*=int(pow(2,max(d-1,0)))
    f=l[i]
    i+=1
a/=factorial(n-f)
print int(a%(10**9+7))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from math import factorial
from bootcamp import Basebootcamp

class Cshaassandlightsbootcamp(Basebootcamp):
    MOD = 10**9 + 7
    MAX_N = 1000  # 根据题目约束限制最大n值
    fact = None
    inv_fact = None

    @classmethod
    def precompute(cls):
        if cls.fact is not None:
            return
        # 预计算阶乘和模逆元
        cls.fact = [1] * (cls.MAX_N + 1)
        for i in range(1, cls.MAX_N + 1):
            cls.fact[i] = cls.fact[i-1] * i % cls.MOD
        
        # 预计算逆元
        cls.inv_fact = [1] * (cls.MAX_N + 1)
        cls.inv_fact[cls.MAX_N] = pow(cls.fact[cls.MAX_N], cls.MOD-2, cls.MOD)
        for i in range(cls.MAX_N-1, -1, -1):
            cls.inv_fact[i] = cls.inv_fact[i+1] * (i+1) % cls.MOD

    def __init__(self, **params):
        # 参数有效性校验
        self.n_min = max(1, min(params.get('n_min', 1), self.MAX_N))
        self.n_max = min(params.get('n_max', self.MAX_N), self.MAX_N)
        
        # 确保n_min <= n_max
        if self.n_min > self.n_max:
            self.n_min, self.n_max = self.n_max, self.n_min
        
        Cshaassandlightsbootcamp.precompute()

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(1, n)
        lights = random.sample(range(1, n+1), m)
        lights.sort()
        return {
            'n': n,
            'm': m,
            'initial_lights': lights
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        lights = question_case['initial_lights']
        lights_str = ' '.join(map(str, lights))
        return f"""你是编程竞赛选手，请解决以下问题：

问题描述：
有{n}盏灯排成一排，编号从1到{n}。初始时有{m}盏灯是亮着的（位置：{lights_str}）。每一步可以选择一个关闭的灯，但前提是该灯必须至少有一个相邻灯已经亮着。每次操作必须立即开启选择的灯，不同的操作顺序视为不同的方式（即使最终所有灯都开启）。例如，灯A和灯B可被按序开启时，顺序A→B和B→A视为两种不同的方式。

计算所有可能的开启顺序的总数，结果对10^9+7取模。

输入格式：
第一行两个整数n和m，第二行m个升序排列的整数表示初始亮灯位置。

当前输入样例：
{n} {m}
{lights_str}

输出要求：
将最终答案放在[answer]和[/answer]标签之间，例如：[answer]42[/answer]。"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def compute_correct_answer(cls, n, m, initial_lights):
        l = sorted(initial_lights)
        total_off = n - m
        result = cls.fact[total_off]
        
        prev = 0
        for i in range(len(l)):
            current = l[i]
            gap = current - prev - 1
            result = result * cls.inv_fact[gap] % cls.MOD
            
            if i > 0:
                result = result * pow(2, max(gap-1, 0), cls.MOD) % cls.MOD
            
            prev = current
        
        last_gap = n - prev
        result = result * cls.inv_fact[last_gap] % cls.MOD
        return result

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            submitted = int(solution)
        except ValueError:
            return False
        
        # 获取题目参数
        n = identity['n']
        m = identity['m']
        lights = identity['initial_lights']
        
        # 计算正确答案
        correct = cls.compute_correct_answer(n, m, lights)
        return (submitted - correct) % cls.MOD == 0
