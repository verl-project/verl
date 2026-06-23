"""# 

### 谜题描述
For a vector \vec{v} = (x, y), define |v| = √{x^2 + y^2}.

Allen had a bit too much to drink at the bar, which is at the origin. There are n vectors \vec{v_1}, \vec{v_2}, ⋅⋅⋅, \vec{v_n}. Allen will make n moves. As Allen's sense of direction is impaired, during the i-th move he will either move in the direction \vec{v_i} or -\vec{v_i}. In other words, if his position is currently p = (x, y), he will either move to p + \vec{v_i} or p - \vec{v_i}.

Allen doesn't want to wander too far from home (which happens to also be the bar). You need to help him figure out a sequence of moves (a sequence of signs for the vectors) such that his final position p satisfies |p| ≤ 1.5 ⋅ 10^6 so that he can stay safe.

Input

The first line contains a single integer n (1 ≤ n ≤ 10^5) — the number of moves.

Each of the following lines contains two space-separated integers x_i and y_i, meaning that \vec{v_i} = (x_i, y_i). We have that |v_i| ≤ 10^6 for all i.

Output

Output a single line containing n integers c_1, c_2, ⋅⋅⋅, c_n, each of which is either 1 or -1. Your solution is correct if the value of p = ∑_{i = 1}^n c_i \vec{v_i}, satisfies |p| ≤ 1.5 ⋅ 10^6.

It can be shown that a solution always exists under the given constraints.

Examples

Input

3
999999 0
0 999999
999999 0


Output

1 1 -1 


Input

1
-824590 246031


Output

1 


Input

8
-67761 603277
640586 -396671
46147 -122580
569609 -2112
400 914208
131792 309779
-850150 -486293
5272 721899


Output

1 1 1 1 1 1 1 -1 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# randomized greedy solution

from sys import stdin, stdout
from random import seed, shuffle

seed(0x5ad5eed)

n = input()
inp = stdin.readlines()

pts = [0 for i in xrange(n)]
pos = [i for i in xrange(n)]
ans = [0 for i in xrange(n)]

i = 0
for line in inp:
    pts[i] = tuple(map(int, line.split()))
    i += 1

R = 1500000 ** 2

while True:
    X, Y = 0, 0
    
    for i in pos:
        x1, y1 = X + pts[i][0], Y + pts[i][1]
        x2, y2 = X - pts[i][0], Y - pts[i][1]

        if (x1 * x1 + y1 * y1 < x2 * x2 + y2 * y2):
            ans[i] = 1
            X, Y = x1, y1
        else:
            ans[i] = -1
            X, Y = x2, y2
        
    if X * X + Y * Y <= R: break

    shuffle(pos)
        
out = [str(ans[i]) for i in xrange(n)]
stdout.write(' '.join(out))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
import re
from bootcamp import Basebootcamp

class Cleavingthebarbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=1000, max_vector_length=10**6):
        self.min_n = min_n
        self.max_n = max_n
        self.max_vector_length = max_vector_length
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        vectors = []
        max_len = self.max_vector_length
        for _ in range(n):
            # 随机选择轴对齐或任意方向
            if random.random() < 0.3:  # 30%概率生成轴对齐向量
                axis = random.choice(['x', 'y'])
                sign = random.choice([1, -1])
                r = random.randint(0, max_len)
                vec = (sign*r, 0) if axis == 'x' else (0, sign*r)
            else:  # 70%概率生成任意方向向量
                while True:
                    x = random.randint(-max_len, max_len)
                    y = random.randint(-max_len, max_len)
                    if x**2 + y**2 <= max_len**2:
                        break
                vec = (x, y)
            vectors.append(vec)
        return {'n': n, 'vectors': vectors}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        vectors = question_case['vectors']
        vectors_str = '\n'.join(f"{x} {y}" for x, y in vectors)
        return f"""Allen醉后需要从酒吧原点出发完成{n}次移动，每次沿±向量方向移动。请选择移动方向使得最终位置距离原点不超过1.5×10^6。

输入格式：
第一行为n
接下来n行每行两个整数x_i y_i

输入：
{n}
{vectors_str}

输出要求：
一行n个1/-1，1表示沿原向量方向，-1表示反向
答案请包裹在[answer]标签内，例如：[answer]1 -1 1 ...[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*([1\s-]+)\s*\[/answer\]', output)
        if not matches:
            return None
        try:
            return list(map(int, matches[-1].strip().split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if len(solution) != identity['n']:
            return False
        x_total, y_total = 0, 0
        for sign, (dx, dy) in zip(solution, identity['vectors']):
            x_total += sign * dx
            y_total += sign * dy
        return math.hypot(x_total, y_total) <= 1.5e6
