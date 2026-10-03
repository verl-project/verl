"""# 

### 谜题描述
Last year the world's largest square was built in Berland. It is known that the square can be represented as an infinite plane with an introduced Cartesian system of coordinates. On that square two sets of concentric circles were painted. Let's call the set of concentric circles with radii 1, 2, ..., K and the center in the point (z, 0) a (K, z)-set. Thus, on the square were painted a (N, x)-set and a (M, y)-set. You have to find out how many parts those sets divided the square into.

Input

The first line contains integers N, x, M, y. (1 ≤ N, M ≤ 100000, - 100000 ≤ x, y ≤ 100000, x ≠ y).

Output

Print the sought number of parts.

Examples

Input

1 0 1 1


Output

4


Input

1 0 1 2


Output

3


Input

3 3 4 7


Output

17

Note

Picture for the third sample:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, x, m, y = map(int, raw_input().split())
if x > y:
	x, y, n, m = y, x, m, n
r, i = n + 1, 1
while i <= m:
	ly, ry = y - i - x, y + i - x
	if ly >= n or ly <= -n:
		r += 1
	elif ly >= 0 and ry >= 0:
		ry = min(n, ry - 1)
		r += 2 * (ry - ly)
	else:
		ry = min(n, ry - 1)
		r += 2 * (ry + ly) + 1
	i += 1
print r
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cberlandsquarebootcamp(Basebootcamp):
    def __init__(self, max_n=1000, max_m=1000, x_range=(-100000, 100000), y_range=(-100000, 100000)):
        self.max_n = max_n
        self.max_m = max_m
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
    
    def case_generator(self):
        # 确保x和y不同
        while True:
            x = random.randint(self.x_min, self.x_max)
            y = random.randint(self.y_min, self.y_max)
            if x != y:
                break
        return {
            "N": random.randint(1, self.max_n),
            "x": x,
            "M": random.randint(1, self.max_m),
            "y": y
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        # 关键修正：使用正确的变量名
        params = {
            "N": question_case["N"],
            "x": question_case["x"],
            "M": question_case["M"],
            "y": question_case["y"]
        }
        return f"""在平面直角坐标系中，有两个由同心圆组成的集合。一个(N, x)-集合由中心在(x, 0)、半径从1到N的圆构成，另一个(M, y)-集合由中心在(y, 0)、半径从1到M的圆构成。这些同心圆将平面划分为多个区域。

你的任务是计算这两个集合的所有圆共同将平面分成多少区域。

输入参数：
- N = {params['N']}
- x = {params['x']}
- M = {params['M']}
- y = {params['y']}

输出要求：
返回一个整数，表示区域的总数。答案必须放置在[answer]和[/answer]之间。

示例：
输入参数1 0 1 1时，输出应为4（两个圆相交形成4个区域）。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 直接调用参考算法验证
        try:
            expected = cls._ref_algorithm(
                identity["N"], identity["x"],
                identity["M"], identity["y"]
            )
            return solution == expected
        except:
            return False

    @staticmethod
    def _ref_algorithm(N, x, M, y):
        """ 参考算法的Python实现 """
        if x > y:
            x, y = y, x
            N, M = M, N
        res = N + 1
        i = 1
        while i <= M:
            ly = y - i - x
            ry = y + i - x
            if ly >= N or ly <= -N:
                res += 1
            else:
                if ly >= 0 and ry >= 0:
                    ry = min(N, ry - 1)
                    res += 2 * (ry - ly)
                else:
                    ry = min(N, ry - 1)
                    res += 2 * (ry + ly) + 1
            i += 1
        return res
