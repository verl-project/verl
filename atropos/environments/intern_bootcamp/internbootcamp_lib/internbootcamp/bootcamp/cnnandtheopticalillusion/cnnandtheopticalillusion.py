"""# 

### 谜题描述
NN is an experienced internet user and that means he spends a lot of time on the social media. Once he found the following image on the Net, which asked him to compare the sizes of inner circles:

<image>

It turned out that the circles are equal. NN was very surprised by this fact, so he decided to create a similar picture himself.

He managed to calculate the number of outer circles n and the radius of the inner circle r. NN thinks that, using this information, you can exactly determine the radius of the outer circles R so that the inner circle touches all of the outer ones externally and each pair of neighboring outer circles also touches each other. While NN tried very hard to guess the required radius, he didn't manage to do that. 

Help NN find the required radius for building the required picture.

Input

The first and the only line of the input file contains two numbers n and r (3 ≤ n ≤ 100, 1 ≤ r ≤ 100) — the number of the outer circles and the radius of the inner circle respectively.

Output

Output a single number R — the radius of the outer circle required for building the required picture. 

Your answer will be accepted if its relative or absolute error does not exceed 10^{-6}.

Formally, if your answer is a and the jury's answer is b. Your answer is accepted if and only when (|a-b|)/(max(1, |b|)) ≤ 10^{-6}.

Examples

Input

3 1


Output

6.4641016


Input

6 1


Output

1.0000000


Input

100 100


Output

3.2429391

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,R = map(int, raw_input().split())

import math
theta = math.pi/n
r = (R*math.sin(theta))/(1-math.sin(theta))
print \"%.7f\"%r
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
import re
from bootcamp import Basebootcamp

class Cnnandtheopticalillusionbootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=100, r_min=1.0, r_max=100.0):
        self.n_min = n_min
        self.n_max = n_max
        self.r_min = r_min
        self.r_max = r_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        # 生成并保留7位小数精度
        r = round(random.uniform(self.r_min, self.r_max), 7)
        return {'n': n, 'r': r}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        r = question_case['r']
        # 保持7位完整小数显示
        return f"""几何构造问题：
- 中心圆半径 r = {r:.7f}
- 外围圆数量 n = {n}

计算外圆半径R，要求：
1. 中心圆与外圆外切
2. 相邻外圆相切
3. 答案保留7位小数

将最终结果放在[answer]标签内，如：[answer]3.1415926[/answer]"""
    
    @staticmethod
    def extract_output(output):
        # 增强匹配模式（支持科学计数法与符号）
        pattern = r'\[answer\]\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\[/answer\]'
        matches = re.findall(pattern, output)
        try:
            return float(matches[-1].replace(',', '')) if matches else None
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None or solution <= 0:  # 答案必须为正数
            return False
        
        n = identity['n']
        r = identity['r']
        theta = math.pi / n
        R_true = (r * math.sin(theta)) / (1 - math.sin(theta))
        
        # 相对/绝对误差校验
        return abs(solution - R_true) <= max(1e-6, 1e-6 * abs(R_true))

# 验证示例
if __name__ == "__main__":
    # 测试小数精度一致性
    bootcamp = Cnnandtheopticalillusionbootcamp(r_min=1.2345678, r_max=1.2345678)
    case = bootcamp.case_generator()
    print(f"案例一致性测试: {case['r'] == 1.2345678}")  # 应输出True

    # 测试负值处理
    print("负值验证:", Cnnandtheopticalillusionbootcamp._verify_correction(-1.0, {'n':3, 'r':1.0}))  # 应返回False

    # 测试科学计数法解析
    print("科学计数法解析:", Cnnandtheopticalillusionbootcamp.extract_output("答案[answer]1.234e-3[/answer]"))  # 应输出0.001234
