"""# 

### 谜题描述
Two villages are separated by a river that flows from the north to the south. The villagers want to build a bridge across the river to make it easier to move across the villages.

The river banks can be assumed to be vertical straight lines x = a and x = b (0 < a < b).

The west village lies in a steppe at point O = (0, 0). There are n pathways leading from the village to the river, they end at points Ai = (a, yi). The villagers there are plain and simple, so their pathways are straight segments as well.

The east village has reserved and cunning people. Their village is in the forest on the east bank of the river, but its exact position is not clear. There are m twisted paths leading from this village to the river and ending at points Bi = (b, y'i). The lengths of all these paths are known, the length of the path that leads from the eastern village to point Bi, equals li.

The villagers want to choose exactly one point on the left bank of river Ai, exactly one point on the right bank Bj and connect them by a straight-line bridge so as to make the total distance between the villages (the sum of |OAi| + |AiBj| + lj, where |XY| is the Euclidean distance between points X and Y) were minimum. The Euclidean distance between points (x1, y1) and (x2, y2) equals <image>.

Help them and find the required pair of points.

Input

The first line contains integers n, m, a, b (1 ≤ n, m ≤ 105, 0 < a < b < 106). 

The second line contains n integers in the ascending order: the i-th integer determines the coordinate of point Ai and equals yi (|yi| ≤ 106). 

The third line contains m integers in the ascending order: the i-th integer determines the coordinate of point Bi and equals y'i (|y'i| ≤ 106). 

The fourth line contains m more integers: the i-th of them determines the length of the path that connects the eastern village and point Bi, and equals li (1 ≤ li ≤ 106).

It is guaranteed, that there is such a point C with abscissa at least b, that |BiC| ≤ li for all i (1 ≤ i ≤ m). It is guaranteed that no two points Ai coincide. It is guaranteed that no two points Bi coincide.

Output

Print two integers — the numbers of points on the left (west) and right (east) banks, respectively, between which you need to build a bridge. You can assume that the points on the west bank are numbered from 1 to n, in the order in which they are given in the input. Similarly, the points on the east bank are numbered from 1 to m in the order in which they are given in the input.

If there are multiple solutions, print any of them. The solution will be accepted if the final length of the path will differ from the answer of the jury by no more than 10 - 6 in absolute or relative value.

Examples

Input

3 2 3 5
-2 -1 4
-1 2
7 3


Output

2 2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from bisect import *
from math import hypot
def main():
    n, m, a, b = map(int, stdin.readline().split())
    A = map(int, stdin.readline().split())
    B = map(int, stdin.readline().split())
    l = map(int, stdin.readline().split())
    Ap = [(x, i) for i, x in enumerate(A)]
    Ap.sort()
    A.sort()
    da = [hypot(x, a) for x in A]
    ansl = 10 ** 9
    ansp = [-1, -1]
    d = b - a
    for i in xrange(m):
        ay = 1.0 * B[i] * a / b
        p = bisect(A, ay)
        for k in xrange(max(0, p-1), min(p+2, n)):
            tmp = da[k] + hypot(A[k] - B[i], d)
            if tmp + l[i] < ansl:
                ansl = tmp + l[i]
                ansp = [Ap[k][1], i]
    print ansp[0] + 1, ansp[1] + 1
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from math import hypot
import bisect
import random
import re

class Dbuildingbridgebootcamp(Basebootcamp):
    def __init__(self, a_range=(1, 1000), b_shift_range=(1, 1000), 
                 n_range=(1, 5), m_range=(1, 5), y_range=(-1000, 1000)):
        self.a_min, self.a_max = a_range
        self.b_shift_min, self.b_shift_max = b_shift_range
        self.n_range = n_range
        self.m_range = m_range
        self.y_min, self.y_max = y_range

    def case_generator(self):
        # 生成满足 0 < a < b <1e6 的整数坐标
        a = random.randint(self.a_min, self.a_max)
        b = a + random.randint(self.b_shift_min, self.b_shift_max)
        while b >= 1e6:
            a = random.randint(1, 999_999)
            b = a + random.randint(1, 999_999 - a)
        
        # 生成西岸点 (已排序的n个唯一整数)
        west_ys = sorted(random.sample(
            range(self.y_min, self.y_max+1),
            random.randint(*self.n_range)
        ))
        
        # 生成东岸点 (已排序的m个唯一整数)
        east_ys = sorted(random.sample(
            range(self.y_min, self.y_max+1),
            random.randint(*self.m_range)
        ))
        
        # 生成保证条件的l_list
        cx = random.randint(b + 1, 1_000_000)  # 东岸村庄x坐标
        cy = random.randint(-1_000_000, 1_000_000)
        l_list = []
        for y in east_ys:
            min_dist = hypot(cx - b, cy - y)
            lj = random.randint(
                max(1, int(min_dist)), 
                max(2, int(min_dist) + 1000)
            )
            l_list.append(lj)
        
        # 计算最优解
        solution, min_total = self.compute_optimal_solution(
            a, b, west_ys, east_ys, l_list
        )
        
        return {
            'n': len(west_ys),
            'm': len(east_ys),
            'a': a,
            'b': b,
            'west_ys': west_ys,
            'east_ys': east_ys,
            'l_list': l_list,
            'solution': solution,
            'min_total': min_total
        }

    @staticmethod
    def compute_optimal_solution(a, b, west_ys, east_ys, l_list):
        delta = b - a
        min_total = float('inf')
        best_pair = (-1, -1)
        
        # 预处理西岸点的索引映射 (输入已排序，索引即为1-based编号)
        indexed_west = list(enumerate(west_ys, 1))
        
        for east_idx, (bj_y, lj) in enumerate(zip(east_ys, l_list), 1):
            # 计算最佳西岸匹配点
            target_y = (bj_y * a) / b
            pos = bisect.bisect_left(west_ys, target_y)
            
            # 检查候选窗口
            candidates = set()
            for offset in (-1, 0, 1):
                k = pos + offset
                if 0 <= k < len(west_ys):
                    candidates.add(k)
            
            # 遍历所有候选点
            for k in candidates:
                ai_y = west_ys[k]
                total = (
                    hypot(a, ai_y) +          # OAi
                    hypot(delta, bj_y - ai_y) + # AiBj
                    lj                        # lj
                )
                if total < min_total:
                    min_total = total
                    best_pair = (k+1, east_idx)  # 转换为1-based索引
        
        return best_pair, min_total

    @staticmethod
    def prompt_func(case):
        question = [
            "Two villages are separated by a river at x={} (west) and x={} (east).".format(case['a'], case['b']),
            "West village paths end at y coordinates (sorted): " + ', '.join(map(str, case['west_ys'])),
            "East village paths end at y coordinates (sorted): " + ', '.join(map(str, case['east_ys'])),
            "East path lengths: " + ', '.join(map(str, case['l_list'])),
            "Find the 1-based indices of optimal west and east points.",
            "Put your answer between [answer] and [/answer], e.g.: [answer]2 3[/answer]"
        ]
        return '\n'.join(question)

    @staticmethod
    def extract_output(output):
        # 支持多种分隔符和首尾空格
        pattern = r'\[answer\]\s*(\d+)[\s,;]+(\d+)\s*\[/answer\]'
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            last_match = matches[-1]
            try:
                return (int(last_match[0]), int(last_match[1]))
            except ValueError:
                pass
        return None

    @classmethod
    def _verify_correction(cls, solution, case):
        if not solution or len(solution) != 2:
            return False
        
        west_idx, east_idx = solution
        west_idx -= 1  # 转换为0-based
        east_idx -= 1
        
        # 索引有效性检查
        if not (0 <= west_idx < case['n'] and 0 <= east_idx < case['m']):
            return False
        
        # 计算实际路径
        a = case['a']
        b = case['b']
        ai_y = case['west_ys'][west_idx]
        bj_y = case['east_ys'][east_idx]
        lj = case['l_list'][east_idx]
        
        total = (
            hypot(a, ai_y) +
            hypot(b - a, bj_y - ai_y) +
            lj
        )
        
        # 允许的误差范围
        return abs(total - case['min_total']) <= 1e-6 * (1 + abs(case['min_total']))
