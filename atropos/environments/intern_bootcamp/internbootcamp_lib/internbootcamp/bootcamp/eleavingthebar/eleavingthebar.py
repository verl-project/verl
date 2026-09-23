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
from sys import stdin, stdout

n = input()
inp = stdin.readlines()

if n == 1:
    print 1
    exit()

a = [0 for i in xrange(3 * n)]

child = [[-1, -1] for i in xrange(len(a))]
sign = [[1, 1] for i in xrange(len(a))]

i = 0
for line in inp:
    a[i] = tuple(map(int, line.split()))
    i += 1

r = 1000000
R = r * r

idx = n
free = [i for i in xrange(n)]

while len(free) >= 3:
    pos = []
    for it in xrange(3): pos.append(free.pop())

    found = 0
    for i in xrange(3):
        for j in xrange(i + 1, 3):
            # |v[i] + v[j]| <= r 
            pt = tuple([a[pos[i]][0] + a[pos[j]][0], a[pos[i]][1] + a[pos[j]][1]])

            if pt[0] * pt[0] + pt[1] * pt[1] <= R:
                child[idx][0] = pos[i]
                child[idx][1] = pos[j]

                sign[idx][0] = 1
                sign[idx][1] = 1

                a[idx] = pt
                free.append(idx)

                free.append(pos[i ^ j ^ 3])

                idx += 1
                found = 1
                break

            # |v[i] - v[j]| <= r
            pt = tuple([a[pos[i]][0] - a[pos[j]][0], a[pos[i]][1] - a[pos[j]][1]])

            if pt[0] * pt[0] + pt[1] * pt[1] <= R:
                child[idx][0] = pos[i]
                child[idx][1] = pos[j]

                sign[idx][0] = 1
                sign[idx][1] = -1

                a[idx] = pt
                free.append(idx)

                free.append(pos[i ^ j ^ 3])

                idx += 1
                found = 1
                break

        if found: break

pos = []
for i in xrange(2): pos.append(free.pop())

pt = tuple([a[pos[0]][0] + a[pos[1]][0], a[pos[0]][1] + a[pos[1]][1]])

if pt[0] * pt[0] + pt[1] * pt[1] <= 2 * R:
    child[idx][0] = pos[0]
    child[idx][1] = pos[1]

    sign[idx][0] = 1
    sign[idx][1] = 1

else:
    child[idx][0] = pos[0]
    child[idx][1] = pos[1]

    sign[idx][0] = 1
    sign[idx][1] = -1

ans = [1 for i in xrange(3 * n)]

stk = [idx]
ptr = 0

while ptr < len(stk):
    u = stk[ptr]
    v1, v2 = child[u]

    if v1 != -1 and v2 != -1:
        stk.append(v1)
        stk.append(v2)

        ans[v1] = sign[u][0] * ans[u]
        ans[v2] = sign[u][1] * ans[u]

    ptr += 1

out = [str(ans[i]) for i in xrange(n)]
stdout.write(' '.join(out))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
import re
from bootcamp import Basebootcamp

class Eleavingthebarbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100, vector_max_magnitude=1e6):
        """
        初始化谜题参数
        
        参数:
            min_n (int): 最小移动次数，默认为1
            max_n (int): 最大移动次数，默认为100
            vector_max_magnitude (float): 单步移动向量的最大模长，默认为1e6
        """
        self.min_n = min_n
        self.max_n = max_n
        self.vector_max_magnitude = vector_max_magnitude
    
    def case_generator(self):
        """
        生成合法谜题实例，保证存在解
        
        返回:
            dict: 包含n和向量列表的可序列化字典
        """
        n = random.randint(self.min_n, self.max_n)
        vectors = []
        max_m = int(self.vector_max_magnitude)
        max_sq = self.vector_max_magnitude ** 2
        
        for _ in range(n):
            x = random.randint(-max_m, max_m)
            x_sq = x ** 2
            remaining = max_sq - x_sq
            if remaining < 0:  # 当x绝对值超过最大值时（理论上不可能）
                y = 0
            else:
                max_y = int(math.sqrt(remaining))
                y = random.randint(-max_y, max_y)
            vectors.append((x, y))
        
        return {
            'n': n,
            'vectors': vectors
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        生成符合格式的系统提示
        
        参数:
            question_case: case_generator生成的谜题实例
            
        返回:
            str: 包含完整问题描述的字符串
        """
        n = question_case['n']
        vectors = question_case['vectors']
        prompt_lines = [
            "Allen喝醉了，正在从位于坐标原点的酒吧出发回家。他需要进行一系列移动，每次可以选择沿着给定方向或反方向移动。",
            "你需要帮助他选择每个移动方向的正负号，使得最终位置与原点的距离不超过1.5×10^6。",
            "\n移动规则说明:",
            "1. 共进行n次移动，每次对应一个二维向量",
            "2. 第i次移动时，Allen可以选择移动方向为c_i·(x_i, y_i)，其中c_i只能是1（正方向）或-1（反方向）",
            "3. 最终位置为所有移动向量的矢量和",
            f"\n当前移动参数 (n={n}):",
            f"{n}"
        ]
        prompt_lines.extend(f"{x} {y}" for x, y in vectors)
        prompt_lines.append("\n请输出包含n个1或-1的序列表示移动方向，格式要求：")
        prompt_lines.append("- 数字之间用空格分隔")
        prompt_lines.append("- 将最终答案放置在[answer]和[/answer]标记之间")
        prompt_lines.append("例如：[answer]1 -1 1[/answer]")
        
        return '\n'.join(prompt_lines)
    
    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取最后一个符合格式的答案
        
        参数:
            output: 模型的完整输出文本
            
        返回:
            list[int] | None: 提取的符号序列（未经验证）
        """
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, flags=re.DOTALL)
        if not matches:
            return None
        
        last_answer = matches[-1].strip()
        try:
            solution = list(map(int, last_answer.split()))
            if all(c in {1, -1} for c in solution):
                return solution
        except ValueError:
            pass
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案的正确性
        
        参数:
            solution: 提取的符号序列
            identity: 谜题实例
            
        返回:
            bool: 是否满足最终位置约束
        """
        n = identity['n']
        vectors = identity['vectors']
        
        # 基础校验
        if len(solution) != n or not all(c in {1, -1} for c in solution):
            return False
        
        # 计算最终位置
        total_x, total_y = 0, 0
        for c, (x, y) in zip(solution, vectors):
            total_x += c * x
            total_y += c * y
        
        # 检查模长约束（平方运算避免浮点误差）
        max_allowed_sq = (1.5e6) ** 2
        total_sq = total_x ** 2 + total_y ** 2
        return total_sq <= max_allowed_sq
