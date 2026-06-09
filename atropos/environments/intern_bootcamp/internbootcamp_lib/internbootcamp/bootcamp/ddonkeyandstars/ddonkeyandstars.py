"""# 

### 谜题描述
In the evenings Donkey would join Shrek to look at the stars. They would sit on a log, sipping tea and they would watch the starry sky. The sky hung above the roof, right behind the chimney. Shrek's stars were to the right of the chimney and the Donkey's stars were to the left. Most days the Donkey would just count the stars, so he knew that they are exactly n. This time he wanted a challenge. He imagined a coordinate system: he put the origin of the coordinates at the intersection of the roof and the chimney, directed the OX axis to the left along the roof and the OY axis — up along the chimney (see figure). The Donkey imagined two rays emanating from he origin of axes at angles α1 and α2 to the OX axis.

<image>

Now he chooses any star that lies strictly between these rays. After that he imagines more rays that emanate from this star at the same angles α1 and α2 to the OX axis and chooses another star that lies strictly between the new rays. He repeats the operation as long as there still are stars he can choose between the rays that emanate from a star. 

<image>

As a result, the Donkey gets a chain of stars. He can consecutively get to each star if he acts by the given rules.

Your task is to find the maximum number of stars m that the Donkey's chain can contain.

Note that the chain must necessarily start in the point of the origin of the axes, that isn't taken into consideration while counting the number m of stars in the chain.

Input

The first line contains an integer n (1 ≤ n ≤ 105) — the number of stars. The second line contains simple fractions representing relationships \"a/b c/d\", such that <image> and <image> (0 ≤ a, b, c, d ≤ 105; <image>; <image>; <image>). The given numbers a, b, c, d are integers.

Next n lines contain pairs of integers xi, yi (1 ≤ xi, yi ≤ 105)— the stars' coordinates.

It is guaranteed that all stars have distinct coordinates.

Output

In a single line print number m — the answer to the problem.

Examples

Input

15
1/3 2/1
3 1
6 2
4 2
2 5
4 5
6 6
3 4
1 6
2 1
7 4
9 3
5 3
1 3
15 5
12 4


Output

4

Note

In the sample the longest chain the Donkey can build consists of four stars. Note that the Donkey can't choose the stars that lie on the rays he imagines.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from bisect import *
n = input()
a, b, c, d = map(int, raw_input().replace(' ', '/').split('/'))
l = []
for x, y in sorted((x, -y) for x, y in [(c * x - d * y, b * y - a * x) for x, y in [map(int, raw_input().split()) for _ in range(n)]] if x > 0 and y > 0):
	v = bisect_left(l, -y)
	if v == len(l):
		l.append(-y)
	else:
		l[v] = -y
print len(l)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bisect import bisect_left
from bootcamp import Basebootcamp
import re
import random
import math

class Ddonkeyandstarsbootcamp(Basebootcamp):
    def __init__(self, max_stars=10, min_param=1, max_param=5):
        """
        参数:
            max_stars: 生成的最大星星数量
            min_param: 角度参数最小取值
            max_param: 角度参数最大取值
        """
        self.max_stars = max_stars
        self.min_param = min_param
        self.max_param = max_param

    def _generate_valid_angles(self):
        """生成满足条件的角度参数: α1 < α2且tan值均为正"""
        while True:
            a, b = random.randint(self.min_param, self.max_param), random.randint(self.min_param, self.max_param)
            c, d = random.randint(self.min_param, self.max_param), random.randint(self.min_param, self.max_param)
            tan1 = a / b
            tan2 = c / d
            if tan1 < tan2 and tan1 > 0 and tan2 > 0:
                return (a, b, c, d)

    def _generate_valid_stars(self, n, a1, b1, c2, d2):
        """生成满足转换后坐标x>0,y>0的星星"""
        stars = []
        for _ in range(n*2):  # 生成冗余数据确保足够有效点
            x = random.randint(1, self.max_stars*2)
            y = random.randint(1, self.max_stars*2)
            # 计算转换后的坐标
            tx = c2 * x - d2 * y
            ty = b1 * y - a1 * x
            if tx > 0 and ty > 0:
                stars.append((x, y))
            if len(stars) >= n:
                break
        return stars[:n]

    def case_generator(self):
        """生成满足约束条件的有效案例"""
        a, b, c, d = self._generate_valid_angles()
        n = random.randint(5, self.max_stars)
        stars = self._generate_valid_stars(n, a, b, c, d)
        
        # 如果有效星星不足，重新生成
        while len(stars) < 3:
            a, b, c, d = self._generate_valid_angles()
            stars = self._generate_valid_stars(n, a, b, c, d)
        
        return {
            'n': len(stars),
            'alpha1': f"{a}/{b}",
            'alpha2': f"{c}/{d}",
            'stars_coordinates': stars,
            '_params': (a, b, c, d)  # 用于验证的内部参数
        }

    @staticmethod
    def prompt_func(question_case):
        alpha1 = question_case['alpha1']
        alpha2 = question_case['alpha2']
        n = question_case['n']
        stars = question_case['stars_coordinates']
        stars_lines = '\n'.join(f"{x} {y}" for x, y in stars)
        
        return f"""You are solving a geometric puzzle. Find the maximum star chain length following these rules:

1. Coordinate system has origin at chimney intersection
2. Initial rays make angles with OX where tan(α1) = {alpha1}, tan(α2) = {alpha2}
3. Each subsequent star must lie strictly between new rays from the previous star
4. Chain starts at origin (not counted)

Input format:
{n}
{alpha1} {alpha2}
{stars_lines}

Output the maximum chain length m. Put your answer between [answer] and [/answer] tags.

Example:
[answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        try:
            # 直接从case参数获取验证参数
            a, b, c, d = identity['_params']
            stars = identity['stars_coordinates']
            
            # 执行参考算法
            transformed = []
            for x, y in stars:
                tx = c * x - d * y
                ty = b * y - a * x
                if tx > 0 and ty > 0:
                    transformed.append((tx, -ty))
            
            transformed.sort()
            lis = []
            for x, y in transformed:
                idx = bisect_left(lis, y)
                if idx == len(lis):
                    lis.append(y)
                else:
                    lis[idx] = y
            return solution == len(lis)
        except:
            return False
