"""# 

### 谜题描述
You have r red, g green and b blue balloons. To decorate a single table for the banquet you need exactly three balloons. Three balloons attached to some table shouldn't have the same color. What maximum number t of tables can be decorated if we know number of balloons of each color?

Your task is to write a program that for given values r, g and b will find the maximum number t of tables, that can be decorated in the required manner.

Input

The single line contains three integers r, g and b (0 ≤ r, g, b ≤ 2·109) — the number of red, green and blue baloons respectively. The numbers are separated by exactly one space.

Output

Print a single integer t — the maximum number of tables that can be decorated in the required manner.

Examples

Input

5 4 3


Output

4


Input

1 1 1


Output

1


Input

2 3 3


Output

2

Note

In the first sample you can decorate the tables with the following balloon sets: \"rgg\", \"gbb\", \"brr\", \"rrg\", where \"r\", \"g\" and \"b\" represent the red, green and blue balls, respectively.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
r,b,g=map(int,raw_input().split())
print min(r+g,g+b,r+b,(r+b+g)/3)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ctabledecorationsbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        支持自定义参数：
        - max_val: 气球数量最大值（默认2e9）
        - min_val: 气球数量最小值（默认0）
        """
        super().__init__(**params)
        self.max_val = params.get('max_val', 2000000000)
        self.min_val = params.get('min_val', 0)
    
    def case_generator(self):
        """生成覆盖各种可能边界的测试案例"""
        # 生成基础随机值
        base_params = {
            'r': random.randint(self.min_val, self.max_val),
            'g': random.randint(self.min_val, self.max_val), 
            'b': random.randint(self.min_val, self.max_val)
        }
        
        # 强制包含特殊边界情况
        special_cases = [
            # 两数极大，一数为0
            {'r': self.max_val, 'g': self.max_val, 'b': 0},
            # 三数相等
            {'r': 100, 'g': 100, 'b': 100},
            # 总和不能被3整除
            {'r': 2, 'g': 2, 'b': 2}
        ]
        
        # 随机选择是否包含特殊案例
        if random.random() < 0.2:  # 20%概率生成特殊案例
            return random.choice(special_cases)
        return base_params
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """增强问题描述的严谨性"""
        r = question_case['r']
        g = question_case['g']
        b = question_case['b']
        return f"""As banquet decorator, you must follow these strict rules:
1. Each table requires EXACTLY 3 balloons
2. All 3 balloons CANNOT be the same color
3. Use exactly {r} red, {g} green, {b} blue balloons

Calculate the MAXIMUM number of tables possible. Format your answer as:

[answer]{{number}}[/answer]

Examples:
Input: 5 4 3 → [answer]4[/answer]
Input: 2 3 3 → [answer]2[/answer]"""

    @staticmethod
    def extract_output(output):
        """增强提取鲁棒性，处理多种格式异常"""
        # 清除可能的换行和空格干扰
        cleaned = output.replace('\n', ' ').replace('\r', '')
        matches = re.findall(r'\[answer\s*](.*?)\[/answer\s*]', cleaned)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """添加类型检查和验证日志"""
        if not isinstance(solution, int):
            return False
            
        r, g, b = identity['r'], identity['g'], identity['b']
        constraints = [
            r + g,
            g + b,
            r + b,
            (r + g + b) // 3
        ]
        return solution == min(constraints)
