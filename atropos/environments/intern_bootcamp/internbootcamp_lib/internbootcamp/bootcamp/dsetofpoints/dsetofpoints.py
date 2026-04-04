"""# 

### 谜题描述
Convexity of a set of points on the plane is the size of the largest subset of points that form a convex polygon. Your task is to build a set of n points with the convexity of exactly m. Your set of points should not contain three points that lie on a straight line.

Input

The single line contains two integers n and m (3 ≤ m ≤ 100, m ≤ n ≤ 2m).

Output

If there is no solution, print \"-1\". Otherwise, print n pairs of integers — the coordinates of points of any set with the convexity of m. The coordinates shouldn't exceed 108 in their absolute value.

Examples

Input

4 3


Output

0 0
3 0
0 3
1 1


Input

6 3


Output

-1


Input

6 6


Output

10 0
-10 0
10 1
9 1
9 -1
0 -2


Input

7 4


Output

176166 6377
709276 539564
654734 174109
910147 434207
790497 366519
606663 21061
859328 886001

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m=map(int,raw_input().split())
print -1 if 3==m and n>4 else '\n'.join(map(lambda x:'%d %d'%x,[(i,i*i) for i in range(m)]+[(i*i+10001,i) for i in range(n-m)]))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dsetofpointsbootcamp(Basebootcamp):
    def __init__(self, m_min=3, m_max=100, max_coordinate=10**8):
        super().__init__()
        self.m_min = max(m_min, 3)  # 确保最小m值符合题目要求
        self.m_max = min(m_max, 100)
        self.max_coordinate = max_coordinate

    def case_generator(self):
        while True:
            m = random.randint(self.m_min, self.m_max)
            max_possible_n = min(2 * m, 100)  # 根据题目约束n ≤ 2m
            
            # 生成合法n的范围
            if m == 3:
                # 允许生成无效案例(n>4)
                n = random.choice([
                    random.randint(3, 4),   # 有效案例
                    random.randint(5, max_possible_n)  # 无效案例
                ])
            else:
                n = random.randint(m, max_possible_n)
            
            # 确保20%概率生成边界案例
            if random.random() < 0.2:
                if m == 3:
                    n = 5  # 强制无效案例
                else:
                    n = 2 * m  # 最大边界案例
            
            return {"n": n, "m": m}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["n"]
        m = question_case["m"]
        example = ""
        if m == 3 and n == 4:
            example = "\n例如当n=4, m=3时，有效解可能是：\n0 0\n3 0\n0 3\n1 1"
        elif m == 6 and n == 6:
            example = "\n例如当n=6, m=6时，有效解可能是：\n10 0\n-10 0\n10 1\n9 1\n9 -1\n0 -2"
        
        prompt = f"""## 平面点集凸度构造问题

### 问题描述
给定两个整数n和m，需要构造一个包含n个点的集合，使得：
1. 集合的凸度（最大凸多边形顶点数）恰好为m
2. 集合中任意三点不共线
3. 所有点坐标绝对值不超过1e8

### 输入参数
- n = {n}
- m = {m}

### 输出要求
{'- 当且仅当m=3且n>4时输出-1' if m ==3 else ''}
- 输出{n}个点的坐标，每行两个整数
- 坐标范围：|x|, |y| ≤ 1e8
{example}

### 答案格式
将答案包裹在[answer]标记中：
[answer]
你的答案
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        # 使用非贪婪匹配查找最后一个答案块
        matches = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_answer = matches[-1].strip()
        if last_answer == "-1":
            return "-1"
        
        try:
            points = [tuple(map(int, line.strip().split())) for line in last_answer.split('\n') if line.strip()]
            return points
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n, m = identity['n'], identity['m']
        
        # 处理无解情况验证
        if solution == "-1":
            return m == 3 and n > 4
        
        # 检查基本格式
        if not isinstance(solution, list) or len(solution) != n:
            return False
        
        # 检查坐标范围和唯一性
        seen = set()
        for x, y in solution:
            if abs(x) > 1e8 or abs(y) > 1e8:
                return False
            if (x, y) in seen:
                return False
            seen.add((x, y))
        
        # 检查三点共线
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = solution[i], solution[j], solution[k]
                    # 计算面积法判断共线
                    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    if area == 0:
                        return False
        
        # 计算凸包
        hull = cls.convex_hull(solution)
        return len(hull) == m

    @staticmethod
    def convex_hull(points):
        """Andrew's monotone chain algorithm"""
        points = sorted(set(points))
        if len(points) <= 1:
            return points
        
        lower = []
        for p in points:
            while len(lower) >= 2:
                a, b = lower[-2], lower[-1]
                cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                if cross > 0:
                    break
                lower.pop()
            lower.append(p)
        
        upper = []
        for p in reversed(points):
            while len(upper) >= 2:
                a, b = upper[-2], upper[-1]
                cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                if cross > 0:
                    break
                upper.pop()
            upper.append(p)
        
        return lower[:-1] + upper[:-1]
