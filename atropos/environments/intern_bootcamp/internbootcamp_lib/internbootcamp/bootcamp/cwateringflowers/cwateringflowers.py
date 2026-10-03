"""# 

### 谜题描述
A flowerbed has many flowers and two fountains.

You can adjust the water pressure and set any values r1(r1 ≥ 0) and r2(r2 ≥ 0), giving the distances at which the water is spread from the first and second fountain respectively. You have to set such r1 and r2 that all the flowers are watered, that is, for each flower, the distance between the flower and the first fountain doesn't exceed r1, or the distance to the second fountain doesn't exceed r2. It's OK if some flowers are watered by both fountains.

You need to decrease the amount of water you need, that is set such r1 and r2 that all the flowers are watered and the r12 + r22 is minimum possible. Find this minimum value.

Input

The first line of the input contains integers n, x1, y1, x2, y2 (1 ≤ n ≤ 2000,  - 107 ≤ x1, y1, x2, y2 ≤ 107) — the number of flowers, the coordinates of the first and the second fountain.

Next follow n lines. The i-th of these lines contains integers xi and yi ( - 107 ≤ xi, yi ≤ 107) — the coordinates of the i-th flower.

It is guaranteed that all n + 2 points in the input are distinct.

Output

Print the minimum possible value r12 + r22. Note, that in this problem optimal answer is always integer.

Examples

Input

2 -1 0 5 3
0 2
5 2


Output

6


Input

4 0 0 5 0
9 4
8 3
-1 0
1 4


Output

33

Note

The first sample is (r12 = 5, r22 = 1): <image> The second sample is (r12 = 1, r22 = 32): <image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, x1, y1, x2, y2 = map(int,raw_input().split())
SquareDist = []
for i in range (0, n):
    x,y = map(int,raw_input().split())
    D1 = (x-x1)**2 + (y-y1)**2
    D2 = (x-x2)**2 + (y-y2)**2
    SquareDist.append((D1,D2))

SquareDist.sort()
Suffixes = [0]
for i in range (0, n):
    Suffixes.append(max(Suffixes[i], SquareDist[n-1-i][1]))

val = Suffixes[n]
for i in range (0, n):
    val = min(val, Suffixes[n-1-i]+SquareDist[i][0])

print val
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re

class Cwateringflowersbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, coord_range=(-100, 100)):  # 修正min_n默认值为1
        """
        初始化参数:
            min_n (int): 生成的最小花朵数量，默认为1以符合题目要求
            max_n (int): 生成的最大花朵数量，默认为10
            coord_range (tuple): 坐标生成范围，默认为(-100, 100)
        """
        self.min_n = min_n
        self.max_n = max_n
        self.coord_range = coord_range
    
    def case_generator(self):
        """
        生成符合要求的谜题实例，确保所有坐标点唯一
        """
        # 生成两个不同的喷泉坐标
        while True:
            x1, y1 = random.randint(*self.coord_range), random.randint(*self.coord_range)
            x2, y2 = random.randint(*self.coord_range), random.randint(*self.coord_range)
            if (x1, y1) != (x2, y2):
                break
        
        n = random.randint(self.min_n, self.max_n)
        existing_points = {(x1, y1), (x2, y2)}
        flowers = []
        
        # 生成n个不重复的flower坐标
        for _ in range(n):
            while True:
                x = random.randint(*self.coord_range)
                y = random.randint(*self.coord_range)
                if (x, y) not in existing_points:
                    existing_points.add((x, y))
                    flowers.append((x, y))
                    break
        
        # 转换为列表形式以便JSON序列化
        fountains = [[x1, y1], [x2, y2]]
        flowers_list = [list(f) for f in flowers]
        
        # 计算所有花的平方距离
        square_dist = []
        for (xi, yi) in flowers:
            d1 = (xi - x1)**2 + (yi - y1)**2
            d2 = (xi - x2)**2 + (yi - y2)**2
            square_dist.append((d1, d2))
        
        # 按照参考算法计算正确解
        square_dist.sort()
        suffixes = [0]
        for i in range(n):
            suffixes.append(max(suffixes[i], square_dist[n-1-i][1]))
        correct_answer = suffixes[-1]
        for i in range(n):
            current_val = square_dist[i][0] + suffixes[n-1-i]
            if current_val < correct_answer:
                correct_answer = current_val
        
        return {
            'n': n,
            'fountains': fountains,
            'flowers': flowers_list,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        fountains = question_case['fountains']
        flowers = question_case['flowers']
        fountain1 = fountains[0]
        fountain2 = fountains[1]
        n = question_case['n']
        
        input_lines = [f"{n} {fountain1[0]} {fountain1[1]} {fountain2[0]} {fountain2[1]}"]
        input_lines.extend(f"{f[0]} {f[1]}" for f in flowers)
        input_str = '\n'.join(input_lines)
        
        flower_coords = '\n'.join(f"({f[0]}, {f[1]})" for f in flowers)
        prompt = f"""你是一个花园管理员，需要优化两个喷泉的覆盖范围。喷泉位于坐标({fountain1[0]}, {fountain1[1]})和({fountain2[0]}, {fountain2[1]})。花园中共有{n}朵花，坐标分别为：
{flower_coords}

你的任务是确定两个非负实数r1和r2，使得所有花朵都被至少一个喷泉覆盖（即每个花朵到喷泉1的距离不超过r1或到喷泉2的距离不超过r2），并且使得r1² + r2²的值最小。

输入格式：
第一行包含n和两个喷泉的坐标，随后n行每行给出花朵坐标。
请输出最小的r1² + r2²的值，并将答案放在[answer]和[/answer]之间。

示例输入：
2 -1 0 5 3
0 2
5 2

示例输出：
6（对应r1²=5，r2²=1）

当前问题输入：
{input_str}

请严格按要求输出整数答案，并放置在指定标签中。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        solution_str = matches[-1].strip()
        try:
            return int(solution_str)
        except ValueError:
            try:
                num = float(solution_str)
                if num.is_integer():
                    return int(num)
            except:
                pass
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
