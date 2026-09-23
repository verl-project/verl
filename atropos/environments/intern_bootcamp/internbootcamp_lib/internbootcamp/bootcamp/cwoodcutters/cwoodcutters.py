"""# 

### 谜题描述
Little Susie listens to fairy tales before bed every day. Today's fairy tale was about wood cutters and the little girl immediately started imagining the choppers cutting wood. She imagined the situation that is described below.

There are n trees located along the road at points with coordinates x1, x2, ..., xn. Each tree has its height hi. Woodcutters can cut down a tree and fell it to the left or to the right. After that it occupies one of the segments [xi - hi, xi] or [xi;xi + hi]. The tree that is not cut down occupies a single point with coordinate xi. Woodcutters can fell a tree if the segment to be occupied by the fallen tree doesn't contain any occupied point. The woodcutters want to process as many trees as possible, so Susie wonders, what is the maximum number of trees to fell. 

Input

The first line contains integer n (1 ≤ n ≤ 105) — the number of trees.

Next n lines contain pairs of integers xi, hi (1 ≤ xi, hi ≤ 109) — the coordinate and the height of the і-th tree.

The pairs are given in the order of ascending xi. No two trees are located at the point with the same coordinate.

Output

Print a single number — the maximum number of trees that you can cut down by the given rules.

Examples

Input

5
1 2
2 1
5 10
10 9
19 1


Output

3


Input

5
1 2
2 1
5 10
10 9
20 1


Output

4

Note

In the first sample you can fell the trees like that: 

  * fell the 1-st tree to the left — now it occupies segment [ - 1;1]
  * fell the 2-nd tree to the right — now it occupies segment [2;3]
  * leave the 3-rd tree — it occupies point 5
  * leave the 4-th tree — it occupies point 10
  * fell the 5-th tree to the right — now it occupies segment [19;20]



In the second sample you can also fell 4-th tree to the right, after that it will occupy segment [10;19].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
x = [0 for i in range(n)]
h = [0 for i in range(n)]
for i in range(n):
	x[i], h[i] = map(int,raw_input().split(\" \"))
if n <= 2:
	print n
else:
	cut = 2
	for i in range(1,n-1):
		if x[i] - h[i] > x[i-1]:
			cut += 1
		elif x[i] + h[i] < x[i+1]:
			x[i] += h[i]
			cut += 1
	print cut
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from copy import deepcopy
from bootcamp import Basebootcamp

class Cwoodcuttersbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, max_h=100, max_x_step=100):
        if min_n < 1:
            raise ValueError("Problem constraint requires n ≥ 1")
        self.min_n = min_n
        self.max_n = max_n
        self.max_h = max_h
        self.max_x_step = max_x_step
    
    def case_generator(self):
        """生成严格递增坐标的树列，确保相邻树间隔≥1"""
        n = random.randint(self.min_n, self.max_n)
        x = []
        current_x = random.randint(1, 10)  # 起始坐标随机
        for _ in range(n):
            x.append(current_x)
            current_x += random.randint(1, self.max_x_step)  # 确保严格递增
        h = [random.randint(1, self.max_h) for _ in range(n)]
        return {'n': n, 'trees': list(map(list, zip(x, h)))}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        trees = question_case['trees']
        input_lines = [f"{x} {h}" for x, h in trees]
        problem_desc = f"""Imagine you are a woodcutter trying to maximize the number of trees cut down under these rules:

1. Each tree can be:
   - Cut to fall left (occupies [{chr(36)}x_i - h_i, {chr(36)}x_i])
   - Cut to fall right (occupies [{chr(36)}x_i, {chr(36)}x_i + h_i])
   - Left standing (occupies point {chr(36)}x_i)
2. Fallen trees' intervals MUST NOT overlap, even at endpoints
3. Trees are given in strictly increasing {chr(36)}x_i order

Input:
{question_case['n']}
{chr(10).join(input_lines)}

What's the maximum number of trees that can be cut? Put ONLY the final number within [answer] tags, like [answer]3[/answer]."""
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(answers[-1]) if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == cls._compute_optimal(identity['trees'])
        except:
            return False
    
    @staticmethod
    def _compute_optimal(trees):
        """严格实现参考代码的贪心算法逻辑"""
        if len(trees) <= 2:
            return len(trees)
        
        x = [t[0] for t in trees]
        h = [t[1] for t in trees]
        x_copy = deepcopy(x)  # 防止修改原始数据
        
        count = 2  # 首尾默认计入
        for i in range(1, len(trees)-1):
            # 优先尝试向左倒
            if x_copy[i] - h[i] > x_copy[i-1]:
                count +=1
            else:
                # 向右倒且不影响下一个
                if x_copy[i] + h[i] < x[i+1]:
                    x_copy[i] += h[i]  # 更新坐标影响后续判断
                    count +=1
        return count
