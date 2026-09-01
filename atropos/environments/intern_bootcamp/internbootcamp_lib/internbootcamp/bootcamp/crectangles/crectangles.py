"""# 

### 谜题描述
You are given n rectangles on a plane with coordinates of their bottom left and upper right points. Some (n-1) of the given n rectangles have some common point. A point belongs to a rectangle if this point is strictly inside the rectangle or belongs to its boundary.

Find any point with integer coordinates that belongs to at least (n-1) given rectangles.

Input

The first line contains a single integer n (2 ≤ n ≤ 132 674) — the number of given rectangles.

Each the next n lines contains four integers x_1, y_1, x_2 and y_2 (-10^9 ≤ x_1 < x_2 ≤ 10^9, -10^9 ≤ y_1 < y_2 ≤ 10^9) — the coordinates of the bottom left and upper right corners of a rectangle.

Output

Print two integers x and y — the coordinates of any point that belongs to at least (n-1) given rectangles.

Examples

Input

3
0 0 1 1
1 1 2 2
3 0 4 1


Output

1 1


Input

3
0 0 1 1
0 1 1 2
1 0 2 1


Output

1 1


Input

4
0 0 5 5
0 0 4 4
1 1 4 4
1 1 4 4


Output

1 1


Input

5
0 0 10 8
1 2 6 7
2 3 5 6
3 4 4 5
8 1 9 2


Output

3 4

Note

The picture below shows the rectangles in the first and second samples. The possible answers are highlighted.

<image>

The picture below shows the rectangles in the third and fourth samples.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=int(raw_input())
ox=[]
cx=[]
l3=[]
oy=[]
cy=[]
for i in range(n):
    r,s,a,b=map(int,raw_input().strip().split())
    l3.append([r,s,a,b])
    ox.append(r)
    oy.append(s)
    cx.append(a)
    cy.append(b)
ox.sort()
cx.sort()
oy.sort()
cy.sort()
e1=ox[-1]
e2=cx[0]
e3=oy[-1]
e4=cy[0]
for i in l3:
    a1=i[0]
    a2=i[1]
    a3=i[2]
    a4=i[3]
    if a1==e1:
        w=ox[-2]
    else:
        w=ox[-1]
    if a2==e3:
        y=oy[-2]
    else:
        y=oy[-1]
    if a3==e2:
        x=cx[1]
    else:
        x=cx[0]
    if a4==e4:
        z=cy[1]
    else:
        z=cy[0]
    if(w<=x and y<=z):
        print w,y
        exit(0)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Crectanglesbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10, coord_range=(-100, 100)):
        self.min_n = min_n
        self.max_n = max_n
        self.coord_range = coord_range
        # 保证公共点生成在中间区域，确保有空间生成不包含的矩形
        self.safe_coord_range = (coord_range[0] + 10, coord_range[1] - 10)

    def case_generator(self):
        """生成保证至少有n-1个矩形交于一点的案例"""
        n = random.randint(self.min_n, self.max_n)
        # 生成公共点（确保不在边界附近）
        x = random.randint(*self.safe_coord_range)
        y = random.randint(*self.safe_coord_range)
        
        rectangles = []
        # 生成前n-1个必然包含公共点的矩形
        for _ in range(n-1):
            x1 = random.randint(self.coord_range[0], x)
            x2 = random.randint(x, self.coord_range[1])
            x1, x2 = sorted([x1, x2])
            # 确保公共点在矩形边界内
            x1 = min(x1, x)
            x2 = max(x2, x)

            y1 = random.randint(self.coord_range[0], y)
            y2 = random.randint(y, self.coord_range[1])
            y1, y2 = sorted([y1, y2])
            y1 = min(y1, y)
            y2 = max(y2, y)
            
            rectangles.append([x1, y1, x2, y2])

        # 生成第n个矩形（可能包含或不包含公共点）
        if random.choice([True, False]):
            # 包含公共点的正常矩形
            x1 = random.randint(self.coord_range[0], x)
            x2 = random.randint(x, self.coord_range[1])
            x1, x2 = sorted([x1, x2])
            x1 = min(x1, x)
            x2 = max(x2, x)

            y1 = random.randint(self.coord_range[0], y)
            y2 = random.randint(y, self.coord_range[1])
            y1, y2 = sorted([y1, y2])
            y1 = min(y1, y)
            y2 = max(y2, y)
        else:
            # 不包含公共点的矩形（确保严格不包含）
            axis = random.choice(['x', 'y'])
            
            # 确保生成有效范围
            if axis == 'x':
                # x轴方向不包含
                direction = random.choice(['left', 'right'])
                if direction == 'left':
                    x_max = x - 1
                    x1 = random.randint(self.coord_range[0], x_max-1)
                    x2 = random.randint(x1+1, x_max)
                else:
                    x_min = x + 1
                    x1 = random.randint(x_min, self.coord_range[1]-1)
                    x2 = random.randint(x1+1, self.coord_range[1])
                # y轴随机生成
                y1 = random.randint(self.coord_range[0], self.coord_range[1]-1)
                y2 = random.randint(y1+1, self.coord_range[1])
            else:
                # y轴方向不包含
                direction = random.choice(['below', 'above'])
                if direction == 'below':
                    y_max = y - 1
                    y1 = random.randint(self.coord_range[0], y_max-1)
                    y2 = random.randint(y1+1, y_max)
                else:
                    y_min = y + 1
                    y1 = random.randint(y_min, self.coord_range[1]-1)
                    y2 = random.randint(y1+1, self.coord_range[1])
                # x轴随机生成
                x1 = random.randint(self.coord_range[0], self.coord_range[1]-1)
                x2 = random.randint(x1+1, self.coord_range[1])
                
            # 二次验证不包含
            assert not (x1 <= x <= x2 and y1 <= y <= y2), "生成错误：矩形包含公共点"
            
        rectangles.append([x1, y1, x2, y2])
        return {'n': n, 'rectangles': rectangles}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        rects = question_case['rectangles']
        problem = f"""给定平面上的{n}个矩形，其中至少{n-1}个有共同点。找到任意属于至少{n-1}个矩形的整数坐标点。

输入格式：
第一行：{n}
接下来{n}行，每行四个整数：x1 y1 x2 y2

具体输入：
{n}
""" + '\n'.join(' '.join(map(str, r)) for r in rects) + """

输出要求：
两个整数x y，用空格分隔，放置在[answer]标签内

示例答案：
[answer]42 314[/answer]"""
        return problem

    @staticmethod
    def extract_output(output):
        # 匹配最后一个出现的答案
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            # 允许多个空格分隔
            x, y = map(int, re.split(r'\s+', last_answer))
            return (x, y)
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        x, y = solution
        count = 0
        for rect in identity['rectangles']:
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                count += 1
        return count >= (identity['n'] - 1)

# 测试代码
if __name__ == "__main__":
    bootcamp = Crectanglesbootcamp()
    for _ in range(3):
        case = bootcamp.case_generator()
        print(f"\n生成的案例：n={case['n']}")
        for r in case['rectangles']:
            print(f"矩形：{r}")
        print(f"正确答案应为包含至少{case['n']-1}个矩形的点，如({bootcamp.safe_coord_range[0]}, {bootcamp.safe_coord_range[1]})")
