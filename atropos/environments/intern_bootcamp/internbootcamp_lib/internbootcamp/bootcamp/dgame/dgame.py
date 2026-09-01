"""# 

### 谜题描述
Allen and Bessie are playing a simple number game. They both know a function f: \{0, 1\}^n → R, i. e. the function takes n binary arguments and returns a real value. At the start of the game, the variables x_1, x_2, ..., x_n are all set to -1. Each round, with equal probability, one of Allen or Bessie gets to make a move. A move consists of picking an i such that x_i = -1 and either setting x_i → 0 or x_i → 1.

After n rounds all variables are set, and the game value resolves to f(x_1, x_2, ..., x_n). Allen wants to maximize the game value, and Bessie wants to minimize it.

Your goal is to help Allen and Bessie find the expected game value! They will play r+1 times though, so between each game, exactly one value of f changes. In other words, between rounds i and i+1 for 1 ≤ i ≤ r, f(z_1, ..., z_n) → g_i for some (z_1, ..., z_n) ∈ \{0, 1\}^n. You are to find the expected game value in the beginning and after each change.

Input

The first line contains two integers n and r (1 ≤ n ≤ 18, 0 ≤ r ≤ 2^{18}).

The next line contains 2^n integers c_0, c_1, ..., c_{2^n-1} (0 ≤ c_i ≤ 10^9), denoting the initial values of f. More specifically, f(x_0, x_1, ..., x_{n-1}) = c_x, if x = \overline{x_{n-1} … x_0} in binary.

Each of the next r lines contains two integers z and g (0 ≤ z ≤ 2^n - 1, 0 ≤ g ≤ 10^9). If z = \overline{z_{n-1} ... z_0} in binary, then this means to set f(z_0, ..., z_{n-1}) → g.

Output

Print r+1 lines, the i-th of which denotes the value of the game f during the i-th round. Your answer must have absolute or relative error within 10^{-6}.

Formally, let your answer be a, and the jury's answer be b. Your answer is considered correct if \frac{|a - b|}{max{(1, |b|)}} ≤ 10^{-6}.

Examples

Input

2 2
0 1 2 3
2 5
0 4


Output

1.500000
2.250000
3.250000


Input

1 0
2 3


Output

2.500000


Input

2 0
1 1 1 1


Output

1.000000

Note

Consider the second test case. If Allen goes first, he will set x_1 → 1, so the final value will be 3. If Bessie goes first, then she will set x_1 → 0 so the final value will be 2. Thus the answer is 2.5.

In the third test case, the game value will always be 1 regardless of Allen and Bessie's play.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from fractions import *
from math import *
from itertools import *
from fractions import*
import string
import copy
import random
import bisect
from decimal import *
from collections import deque
def id_generator(size=20, chars=string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
def mp():
    return map(int,str(raw_input()).split())
n,r=mp()
l=list(mp())
x=pow(2,n)
s=sum(l)
print(s/float(x))
for i in range(r):
	a,b=mp()
	s=s-l[a]
	s+=b
	l[a]=b
	print(s/float(x))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import isclose

from bootcamp import Basebootcamp

class Dgamebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 2)
        self.r = params.get('r', 0)
        # 验证n和r的范围
        if self.n < 1 or self.n > 18:
            raise ValueError("n must be between 1 and 18")
        if self.r < 0 or self.r > (2 ** 18):
            raise ValueError("r must be between 0 and 2^18")
    
    def case_generator(self):
        n = self.n
        r = self.r
        size = 2 ** n
        # 生成初始c数组，考虑边界情况
        initial_c = [random.randint(0, 10**9) for _ in range(size)]
        # 确保至少有一次修改，如果r>0
        updates = []
        for _ in range(r):
            z = random.randint(0, size - 1)
            g = random.randint(0, 10**9)
            updates.append((z, g))
        case = {
            'n': n,
            'initial_c': initial_c,
            'r': r,
            'updates': updates
        }
        return case
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        initial_c = question_case['initial_c']
        r = question_case['r']
        prompt = (
            "Allen和Bessie正在玩一个数字游戏。游戏规则如下：\n"
            "游戏由一个函数f决定，f接受n个二进制变量，返回一个实数值。"
            "初始时，所有变量设为-1，每轮随机选择一个玩家设置一个变量为0或1。"
            "游戏目标是计算在每一轮修改后的期望值，输出r+1个结果，每个结果保留六位小数。\n"
            "初始函数值为："
        )
        prompt += f"{initial_c}\n"
        prompt += "每次修改操作如下：\n"
        for i, (z, g) in enumerate(question_case['updates'], 1):
            prompt += f"第{i}次修改：将f的值{z}修改为{g}\n"
        prompt += (
            "请输出每次修改后的期望值，每个值放在[answer]标签内，"
            "保留六位小数，例如：[answer]1.500000[/answer]。"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output)
        if not matches:
            return None
        solutions = []
        for m in matches:
            m_clean = m.strip()
            # 处理可能的格式问题，例如多余的空格或换行符
            if m_clean == "":
                continue
            try:
                num = float(m_clean)
                solutions.append(num)
            except ValueError:
                pass
        if not solutions:
            return None
        return solutions
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if isinstance(solution, list):
            solutions = solution
        else:
            solutions = [solution]
        n = identity['n']
        size = 2 ** n
        initial_c = identity['initial_c']
        updates = identity['updates']
        r = identity['r']
        correct = []
        current_sum = sum(initial_c)
        correct_value = current_sum / size
        correct.append(correct_value)
        current_c = initial_c.copy()
        for z, g in updates:
            delta = g - current_c[z]
            current_sum += delta
            current_c[z] = g
            correct_value = current_sum / size
            correct.append(correct_value)
        if len(solutions) != len(correct):
            return False
        for s, c in zip(solutions, correct):
            if not isclose(s, c, rel_tol=1e-6, abs_tol=1e-6):
                return False
        return True
