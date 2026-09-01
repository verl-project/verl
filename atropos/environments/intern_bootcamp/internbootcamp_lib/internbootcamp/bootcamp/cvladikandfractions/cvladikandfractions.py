"""# 

### 谜题描述
Vladik and Chloe decided to determine who of them is better at math. Vladik claimed that for any positive integer n he can represent fraction <image> as a sum of three distinct positive fractions in form <image>.

Help Vladik with that, i.e for a given n find three distinct positive integers x, y and z such that <image>. Because Chloe can't check Vladik's answer if the numbers are large, he asks you to print numbers not exceeding 109.

If there is no such answer, print -1.

Input

The single line contains single integer n (1 ≤ n ≤ 104).

Output

If the answer exists, print 3 distinct numbers x, y and z (1 ≤ x, y, z ≤ 109, x ≠ y, x ≠ z, y ≠ z). Otherwise print -1.

If there are multiple answers, print any of them.

Examples

Input

3


Output

2 7 42


Input

7


Output

7 8 56

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
z = 0
if(n==1):
    print -1
else:
    for x in xrange(2,1000000001):
        for y in xrange(x+1,1000000001):
            if((1/float(x)+1/float(y))>=2/float(n)):
                break
            z = float(n*x*y)/float(2*x*y-n*y-n*x)
            if z==float(int(z)) and z!=x and z!=y:
                print x, y, int(z)
                break
            else:
                if(z<y):
                    z = 0
                    break
                z = 0
        if(z>0):
            break
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cvladikandfractionsbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)  # 显式调用基类初始化
        self.max_n = params.get('max_n', 10000)  # 默认上限为10^4
    
    def case_generator(self):
        # 生成n的范围为1到max_n，包含所有可能的问题实例
        # 当n=1时答案应为-1，其他n≥2根据题目逻辑应有解
        n = random.randint(1, self.max_n)
        return {"n": n}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case["n"]
        problem_text = (
            f"Vladik and Chloe are determining who is better at math. Vladik claims that for any positive integer n, "
            f"the fraction 2/n can be expressed as the sum of three distinct positive unit fractions. Help Vladik prove this "
            f"by finding three distinct positive integers x, y, z such that 1/x + 1/y + 1/z = 2/{n}. The numbers x, y, z must "
            f"not exceed 1e9. If it's impossible, output -1.\n\n"
            "Provide your answer as three space-separated integers or -1 enclosed within [answer] and [/answer] tags. "
            "Example: [answer]2 7 42[/answer] or [answer]-1[/answer] if no solution exists."
        )
        return problem_text
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        if last_answer == '-1':
            return -1
        parts = last_answer.split()
        if len(parts) != 3:
            return None
        try:
            x, y, z = map(int, parts)
            return (x, y, z)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity["n"]
        if n == 1:
            return solution == -1  # n=1时只有-1正确
        else:
            # 其他n必须返回有效三元组
            if solution == -1:
                return False  # n≥2时不接受-1
            if not isinstance(solution, tuple) or len(solution) != 3:
                return False
            x, y, z = solution
            if x <= 0 or y <= 0 or z <= 0:
                return False
            if x > 1e9 or y > 1e9 or z > 1e9:
                return False
            if x == y or x == z or y == z:
                return False
            # 关键数学验证：n(xy + yz + zx) == 2xyz
            left = n * (x * y + y * z + z * x)
            right = 2 * x * y * z
            return left == right
