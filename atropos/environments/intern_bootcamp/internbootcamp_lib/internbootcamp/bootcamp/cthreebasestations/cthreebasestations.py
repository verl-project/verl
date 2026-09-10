"""# 

### 谜题描述
The New Vasjuki village is stretched along the motorway and that's why every house on it is characterized by its shift relative to some fixed point — the xi coordinate. The village consists of n houses, the i-th house is located in the point with coordinates of xi.

TELE3, a cellular communication provider planned to locate three base stations so as to provide every house in the village with cellular communication. The base station having power d located in the point t provides with communication all the houses on the segment [t - d, t + d] (including boundaries).

To simplify the integration (and simply not to mix anything up) all the three stations are planned to possess the equal power of d. Which minimal value of d is enough to provide all the houses in the village with cellular communication.

Input

The first line contains an integer n (1 ≤ n ≤ 2·105) which represents the number of houses in the village. The second line contains the coordinates of houses — the sequence x1, x2, ..., xn of integer numbers (1 ≤ xi ≤ 109). It is possible that two or more houses are located on one point. The coordinates are given in a arbitrary order.

Output

Print the required minimal power d. In the second line print three numbers — the possible coordinates of the base stations' location. Print the coordinates with 6 digits after the decimal point. The positions of the stations can be any from 0 to 2·109 inclusively. It is accepted for the base stations to have matching coordinates. If there are many solutions, print any of them.

Examples

Input

4
1 2 3 4


Output

0.500000
1.500000 2.500000 3.500000


Input

3
10 20 30


Output

0
10.000000 20.000000 30.000000


Input

5
10003 10004 10001 10002 1


Output

0.500000
1.000000 10001.500000 10003.500000

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#nealzane
from bisect import bisect_left as L,bisect_right as R
F=lambda x:'%.6lf'%x
n=input()
a=sorted(map(lambda x:int(x)*2,raw_input().split()))
l,h=0,1<<31
while l<h:
  d=(l+h)//2
  s=d*2
  x,y=R(a,a[0]+s),L(a,a[-1]-s)
  if x<y and a[y-1]-a[x]>s:
    l=d+1
  else:
    h=d
print F(l/2.)
x,y=R(a,a[0]+l*2),L(a,a[-1]-l*2)
if x>y:x=y
print ' '.join(map(F,((a[0]+a[x-1])/4.,(a[x]+a[y-1])/4.,(a[y]+a[-1])/4.)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bisect import bisect_left, bisect_right
import random
import re

class Cthreebasestationsbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10**5, min_x=1, max_x=2*10**9):
        self.n_min = max(n_min, 1)  # 确保n≥1
        self.n_max = n_max
        self.min_x = min_x
        self.max_x = max_x

    def case_generator(self):
        # 增加边界情况生成概率
        if random.random() < 0.2:
            n = random.choice([1, 2, 3, 10**5])
        else:
            n = random.randint(self.n_min, self.n_max)
        
        # 生成特殊案例
        if random.random() < 0.15:
            x = random.randint(self.min_x, self.max_x)
            houses = [x] * n  # 所有房屋同一坐标
        else:
            houses = [random.randint(self.min_x, self.max_x) for _ in range(n)]
        
        correct_d, stations = self._compute_solution(n, houses)
        return {
            'n': n,
            'houses': houses,
            'correct_d': correct_d,
            'correct_stations': stations
        }

    @staticmethod
    def _compute_solution(n, houses):
        a = sorted([x * 2 for x in houses])
        if not a:
            return 0.0, [0.0, 0.0, 0.0]
        
        left, right = 0, 1 << 31
        
        # 二分查找最小d
        while left < right:
            mid = (left + right) // 2
            s = mid * 2
            x = bisect_right(a, a[0] + s)
            y = bisect_left(a, a[-1] - s)
            
            if x < y and (a[y-1] - a[x] > s):
                left = mid + 1
            else:
                right = mid
        
        d = left
        correct_d = d / 2.0
        
        # 计算基站坐标
        x_val = bisect_right(a, a[0] + d * 2)
        y_val = bisect_left(a, a[-1] - d * 2)
        
        # 处理全范围覆盖的情况
        if x_val >= len(a):
            return correct_d, [a[0]/2.0, a[0]/2.0, a[0]/2.0]
        
        # 计算三段分割点
        s1 = (a[0] + a[x_val-1])/4.0 if x_val > 0 else a[0]/2.0
        s2 = (a[x_val] + a[y_val-1])/4.0 if x_val < y_val else s1
        s3 = (a[y_val] + a[-1])/4.0 if y_val < len(a) else a[-1]/2.0
        
        return correct_d, [s1, s2, s3]

    @staticmethod
    def prompt_func(question_case):
        houses = question_case['houses']
        return f"""The New Vasjuki village needs to install three base stations with equal power d. All houses must be covered by [t-d, t+d] ranges.

Input:
n = {question_case['n']}
House coordinates (unsorted): {' '.join(map(str, houses))}

Output format:
1. Minimal d with exactly 6 decimal places
2. Three station coordinates with exactly 6 decimal places

Put your final answer between [answer] and [/answer]. Example:
[answer]
0.500000
1.500000 2.500000 3.500000
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 提取最后一个答案块
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        content = answer_blocks[-1].strip().split('\n')
        if len(content) < 2:
            return None
        
        try:
            d = round(float(content[0].strip()), 6)
            stations = [round(float(x.strip()), 6) for x in content[1].split()]
            if len(stations) != 3:
                return None
            return {'d': d, 'stations': stations}
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or 'd' not in solution or 'stations' not in solution:
            return False
        
        solution_d = solution['d']
        epsilon = 1e-7  # 扩大容差范围
        
        # 验证d值精度
        if abs(solution_d - identity['correct_d']) > epsilon:
            return False
        
        # 验证所有房屋被覆盖
        stations = solution['stations']
        for house in identity['houses']:
            if not any(
                (house >= (s - solution_d - epsilon)) and 
                (house <= (s + solution_d + epsilon))
                for s in stations
            ):
                return False
        
        return True
