"""# 

### 谜题描述
Leonid wants to become a glass carver (the person who creates beautiful artworks by cutting the glass). He already has a rectangular w mm  ×  h mm sheet of glass, a diamond glass cutter and lots of enthusiasm. What he lacks is understanding of what to carve and how.

In order not to waste time, he decided to practice the technique of carving. To do this, he makes vertical and horizontal cuts through the entire sheet. This process results in making smaller rectangular fragments of glass. Leonid does not move the newly made glass fragments. In particular, a cut divides each fragment of glass that it goes through into smaller fragments.

After each cut Leonid tries to determine what area the largest of the currently available glass fragments has. Since there appear more and more fragments, this question takes him more and more time and distracts him from the fascinating process.

Leonid offers to divide the labor — he will cut glass, and you will calculate the area of the maximum fragment after each cut. Do you agree?

Input

The first line contains three integers w, h, n (2 ≤ w, h ≤ 200 000, 1 ≤ n ≤ 200 000).

Next n lines contain the descriptions of the cuts. Each description has the form H y or V x. In the first case Leonid makes the horizontal cut at the distance y millimeters (1 ≤ y ≤ h - 1) from the lower edge of the original sheet of glass. In the second case Leonid makes a vertical cut at distance x (1 ≤ x ≤ w - 1) millimeters from the left edge of the original sheet of glass. It is guaranteed that Leonid won't make two identical cuts.

Output

After each cut print on a single line the area of the maximum available glass fragment in mm2.

Examples

Input

4 3 4
H 2
V 2
V 3
V 1


Output

8
4
4
2


Input

7 6 5
H 4
V 3
V 5
H 2
V 1


Output

28
16
12
6
4

Note

Picture for the first sample test: 

<image> Picture for the second sample test:  <image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin, stdout
from itertools import repeat
def main():
    w, h, n = map(int, stdin.readline().split())
    a = [stdin.readline().split() for _ in xrange(n)]
    y = [0, h]
    x = [0, w]
    for i in xrange(n):
        a[i][1] = int(a[i][1], 10)
        if a[i][0] == 'H':
            y.append(a[i][1])
        else:
            x.append(a[i][1])
    y.sort()
    x.sort()
    iy = {t: i for i, t in enumerate(y)}
    ix = {t: i for i, t in enumerate(x)}
    ny = len(y)
    nx = len(x)
    pary = range(len(y))
    parx = range(len(x))
    p = 0
    dy = [0] * ny
    for i in xrange(ny - 1):
        dy[i] = y[i+1] - y[i]
    my = max(dy)
    dx = [0] * nx
    for i in xrange(nx - 1):
        dx[i] = x[i+1] - x[i]
    mx = max(dx)
    ans = [my * mx]
    for t in reversed(a):
        if t[0] == 'H':
            i = iy[t[1]]
            st = [i]
            while pary[i] != i:
                i = pary[i]
                st.append(i)
            nl = dy[i]
            i = iy[t[1]] - 1
            st.append(i)
            while pary[i] != i:
                i = pary[i]
                st.append(i)
            dy[i] += nl
            if my < dy[i]:
                my = dy[i]
            i = st.pop()
            for j in st:
                pary[j] = i
        else:
            i = ix[t[1]]
            st = [i]
            while parx[i] != i:
                i = parx[i]
                st.append(i)
            nl = dx[i]
            i = ix[t[1]] - 1
            st.append(i)
            while parx[i] != i:
                i = parx[i]
                st.append(i)
            dx[i] += nl
            if mx < dx[i]:
                mx = dx[i]
            i = st.pop()
            for j in st:
                parx[j] = i
        ans.append(mx * my)
    ans.pop()
    stdout.write('\n'.join(map(str, reversed(ans))))
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import bisect
import re
from bootcamp import Basebootcamp

class Aglasscarvingbootcamp(Basebootcamp):
    def __init__(self, max_width=1000, max_height=1000, max_cuts=20):
        self.max_width = max_width
        self.max_height = max_height
        self.max_cuts = max_cuts

    def case_generator(self):
        w = random.randint(2, self.max_width)
        h = random.randint(2, self.max_height)
        max_possible = (w - 1) + (h - 1)
        n = random.randint(1, min(self.max_cuts, max_possible))
        
        h_available = set(range(1, h))
        v_available = set(range(1, w))
        cuts = []
        for _ in range(n):
            possible = []
            if h_available:
                possible.append('H')
            if v_available:
                possible.append('V')
            if not possible:
                break
            direction = random.choice(possible)
            if direction == 'H':
                y = random.choice(list(h_available))
                h_available.remove(y)
                cuts.append(('H', y))
            else:
                x = random.choice(list(v_available))
                v_available.remove(x)
                cuts.append(('V', x))
        n_actual = len(cuts)
        x_coords = [0, w]
        y_coords = [0, h]
        max_w = w
        max_h = h
        expected_outputs = []
        for cut in cuts:
            direction, pos = cut
            if direction == 'H':
                bisect.insort(y_coords, pos)
                max_h = max(y_coords[i] - y_coords[i-1] for i in range(1, len(y_coords)))
            else:
                bisect.insort(x_coords, pos)
                max_w = max(x_coords[i] - x_coords[i-1] for i in range(1, len(x_coords)))
            current_area = max_w * max_h
            expected_outputs.append(current_area)
        case = {
            'w': w,
            'h': h,
            'n': n_actual,
            'cuts': cuts,
            'expected_outputs': expected_outputs
        }
        return case

    @staticmethod
    def prompt_func(question_case):
        w = question_case['w']
        h = question_case['h']
        n = question_case['n']
        cuts = question_case['cuts']
        cuts_str = "\n".join([f"{d} {pos}" for d, pos in cuts])
        prompt = f"""Leonid正在练习玻璃切割技术，每次切割后需要计算当前最大的玻璃碎片面积。你的任务是帮助他在每次切割后输出对应的最大面积。

题目规则：

1. 初始玻璃片尺寸为{w}毫米（宽） × {h}毫米（高）。
2. 依次进行{n}次切割，每次切割为垂直（V）或水平（H）方向，切穿整个玻璃片。每个切割的位置不会重复。
3. 每次切割后，你需要立即输出当前存在的所有玻璃碎片中的最大面积（单位：平方毫米）。

输入格式：

第一行包含三个整数：W H N，分别表示初始宽度、高度和切割次数。
接下来的N行每行格式为'H Y'或'V X'，其中Y是距离下边缘的毫米数，X是距离左边缘的毫米数。

现在，请解决以下问题：

初始宽度：{w}
初始高度：{h}
切割次数：{n}
切割列表：
{cuts_str}

请按照切割顺序，每次切割后输出对应的最大面积，并将所有结果按顺序放在[answer]和[/answer]标签之间，每个结果占一行。"""
        return prompt

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        lines = last_block.split('\n')
        answers = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    num = int(line)
                    answers.append(num)
                except:
                    pass
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_outputs']
        return solution == expected if solution is not None else False
