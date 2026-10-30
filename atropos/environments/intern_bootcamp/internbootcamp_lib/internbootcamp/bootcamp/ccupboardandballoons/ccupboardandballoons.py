"""# 

### 谜题描述
A girl named Xenia has a cupboard that looks like an arc from ahead. The arc is made of a semicircle with radius r (the cupboard's top) and two walls of height h (the cupboard's sides). The cupboard's depth is r, that is, it looks like a rectangle with base r and height h + r from the sides. The figure below shows what the cupboard looks like (the front view is on the left, the side view is on the right).

<image>

Xenia got lots of balloons for her birthday. The girl hates the mess, so she wants to store the balloons in the cupboard. Luckily, each balloon is a sphere with radius <image>. Help Xenia calculate the maximum number of balloons she can put in her cupboard. 

You can say that a balloon is in the cupboard if you can't see any part of the balloon on the left or right view. The balloons in the cupboard can touch each other. It is not allowed to squeeze the balloons or deform them in any way. You can assume that the cupboard's walls are negligibly thin.

Input

The single line contains two integers r, h (1 ≤ r, h ≤ 107).

Output

Print a single integer — the maximum number of balloons Xenia can put in the cupboard.

Examples

Input

1 1


Output

3


Input

1 2


Output

5


Input

2 1


Output

2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from math import sqrt
r ,h = map(int,raw_input().split());
u = h % r;
ans = 2*(h/r);
if 2 * u >= sqrt(3) * r:
	ans += 3;
elif 2*u >= r:
	ans += 2;
else:
	ans += 1;

print ans;
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import sqrt
from bootcamp import Basebootcamp

class Ccupboardandballoonsbootcamp(Basebootcamp):
    def __init__(self, r_min=1, r_max=10**7, h_min=1, h_max=10**7):
        """新增参数联动校验"""
        # 保证r参数有效性
        if not (1 <= r_min <= r_max):
            raise ValueError(f"Invalid r range: [{r_min}, {r_max}]")
        # 保证h参数有效性
        if not (1 <= h_min <= h_max):
            raise ValueError(f"Invalid h range: [{h_min}, {h_max}]")
        
        self._validate_params(r_min, r_max, h_min, h_max)
        
        self.r_min = r_min
        self.r_max = r_max
        self.h_min = h_min
        self.h_max = h_max

    def _validate_params(self, r_min, r_max, h_min, h_max):
        """确保案例生成的可行性"""
        if r_min > h_max:
            raise ValueError(f"r_min({r_min}) cannot exceed h_max({h_max}) for integer multiple cases")

    def case_generator(self):
        """重构案例生成逻辑"""
        rand_type = random.random()
        
        # 处理分层案例生成
        if rand_type < 0.5:
            return self._generate_common_case()
        elif rand_type < 0.8:
            return self._generate_integer_case()
        else:
            return self._generate_boundary_case()

    def _generate_common_case(self):
        """生成普通随机案例"""
        return {
            'r': random.randint(self.r_min, self.r_max),
            'h': random.randint(self.h_min, self.h_max)
        }

    def _generate_integer_case(self):
        """生成h为r整数倍的案例"""
        possible_r_max = min(self.r_max, self.h_max)
        if self.r_min > possible_r_max:
            return self._generate_common_case()
        
        r = random.randint(self.r_min, possible_r_max)
        max_k = self.h_max // r
        if max_k < 1:
            return self._generate_common_case()
        
        return {
            'r': r,
            'h': r * random.randint(1, max_k)
        }

    def _generate_boundary_case(self):
        """生成边界值组合案例"""
        return {
            'r': random.choice([self.r_min, self.r_max]),
            'h': random.choice([self.h_min, self.h_max])
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        r = question_case['r']
        h = question_case['h']
        return f"""Xenia的柜子结构：
- 正面由半圆（半径{r}米）和垂直侧壁（高{h}米）组成
- 侧面可视深度{r}米，总高度{h + r}米
- 每个气球为半径{r}米的完美球体

放置规则：
1. 气球需完全包含在柜体内（左右视图不可见）
2. 允许接触但不可重叠或变形
3. 柜壁厚度忽略

当前参数：r = {r}, h = {h}

请计算最大容纳气球数，并将整数答案放在[answer]标签内，例如：[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强鲁棒性匹配
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        r = identity['r']
        h = identity['h']
        
        # 核心验证逻辑
        layers = h // r
        remainder = h % r
        
        base = 2 * layers
        if 2*remainder >= r * sqrt(3):
            base += 3
        elif 2*remainder >= r:
            base += 2
        else:
            base += 1
            
        try:
            return int(solution) == base
        except (ValueError, TypeError):
            return False
