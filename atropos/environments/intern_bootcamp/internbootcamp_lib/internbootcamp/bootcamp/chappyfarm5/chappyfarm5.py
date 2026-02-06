"""# 

### 谜题描述
The Happy Farm 5 creators decided to invent the mechanism of cow grazing. The cows in the game are very slow and they move very slowly, it can even be considered that they stand still. However, carnivores should always be chased off them. 

For that a young player Vasya decided to make the shepherd run round the cows along one and the same closed path. It is very important that the cows stayed strictly inside the area limited by the path, as otherwise some cows will sooner or later be eaten. To be absolutely sure in the cows' safety, Vasya wants the path completion time to be minimum.

The new game is launched for different devices, including mobile phones. That's why the developers decided to quit using the arithmetics with the floating decimal point and use only the arithmetics of integers. The cows and the shepherd in the game are represented as points on the plane with integer coordinates. The playing time is modeled by the turns. During every turn the shepherd can either stay where he stands or step in one of eight directions: horizontally, vertically, or diagonally. As the coordinates should always remain integer, then the length of a horizontal and vertical step is equal to 1, and the length of a diagonal step is equal to <image>. The cows do not move. You have to minimize the number of moves the shepherd needs to run round the whole herd.

Input

The first line contains an integer N which represents the number of cows in the herd (1 ≤ N ≤ 105). Each of the next N lines contains two integers Xi and Yi which represent the coordinates of one cow of (|Xi|, |Yi| ≤ 106). Several cows can stand on one point.

Output

Print the single number — the minimum number of moves in the sought path.

Examples

Input

4
1 1
5 1
5 3
1 3


Output

16

Note

Picture for the example test: The coordinate grid is painted grey, the coordinates axes are painted black, the cows are painted red and the sought route is painted green.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
a = b = c = d = -6666666
for i in range(n):
	x, y = map(int, raw_input().split())
	a = max(a, x + y)
	b = max(b, x - y)
	c = max(c, y - x)
	d = max(d, - x - y)
print a + b + c + d + 4
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Chappyfarm5bootcamp(Basebootcamp):
    def __init__(self, n=4, max_a=100):
        """
        初始化训练场参数
        :param n: 牛的数量，默认4
        :param max_a: 坐标最大生成范围，默认100
        """
        self.n = n
        self.max_a = max_a
    
    def case_generator(self):
        # 生成核心参数
        a = random.randint(0, self.max_a)
        b = random.randint(0, a)
        c = random.randint(0, a)
        d = random.randint(0, b)
        
        # 构造核心点
        points = []
        # Point for a = x + y
        x_a = a // 2
        y_a = a - x_a
        points.append((x_a, y_a))
        
        # Point for b = x - y
        points.append((b, 0))
        
        # Point for c = y - x
        points.append((0, c))
        
        # Point for d = -x - y
        points.append((0, -d))
        
        # 生成额外点（如果n>4）
        required = self.n - 4
        if required > 0:
            count = 0
            max_attempts = 1000  # 防止无限循环
            while count < required and max_attempts > 0:
                max_attempts -= 1
                xi = random.randint(-self.max_a, self.max_a)
                yi = random.randint(-self.max_a, self.max_a)
                if (xi + yi <= a and
                    xi - yi <= b and
                    yi - xi <= c and
                    (-xi - yi) <= d):
                    points.append((xi, yi))
                    count += 1
        
        # 构造案例字典
        cows = [list(p) for p in points]
        expected = a + b + c + d + 4
        
        return {
            "n": len(cows),
            "cows": cows,
            "expected": expected
        }
    
    @staticmethod
    def prompt_func(question_case):
        cows = question_case['cows']
        cows_str = '\n'.join(f"{x} {y}" for x, y in cows)
        
        prompt = f"""你正在玩Happy Farm游戏，需要帮助牧羊人找到围绕所有牛群的最短闭合路径。路径必须满足以下条件：
1. 所有牛必须严格位于路径内部
2. 路径必须闭合（回到起点）
3. 牧羊人每步可以移动到相邻的8个方向或保持不动

输入数据：
第一行包含牛的数量N
接下来N行每行包含两个整数，表示牛的坐标

当前问题：
N = {question_case['n']}
坐标列表：
{cows_str}

请计算最少需要多少步完成这个闭合路径，并将答案放入[answer]标签中。例如：[answer]16[/answer]"""

        return prompt
    
    @staticmethod
    def extract_output(output):
        # 提取最后一个[answer]标签内容
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['expected']
        except (ValueError, TypeError):
            return False
