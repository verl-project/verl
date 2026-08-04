"""# 

### 谜题描述
Dima's got a staircase that consists of n stairs. The first stair is at height a1, the second one is at a2, the last one is at an (1 ≤ a1 ≤ a2 ≤ ... ≤ an). 

Dima decided to play with the staircase, so he is throwing rectangular boxes at the staircase from above. The i-th box has width wi and height hi. Dima throws each box vertically down on the first wi stairs of the staircase, that is, the box covers stairs with numbers 1, 2, ..., wi. Each thrown box flies vertically down until at least one of the two following events happen:

  * the bottom of the box touches the top of a stair; 
  * the bottom of the box touches the top of a box, thrown earlier. 



We only consider touching of the horizontal sides of stairs and boxes, at that touching with the corners isn't taken into consideration. Specifically, that implies that a box with width wi cannot touch the stair number wi + 1.

You are given the description of the staircase and the sequence in which Dima threw the boxes at it. For each box, determine how high the bottom of the box after landing will be. Consider a box to fall after the previous one lands.

Input

The first line contains integer n (1 ≤ n ≤ 105) — the number of stairs in the staircase. The second line contains a non-decreasing sequence, consisting of n integers, a1, a2, ..., an (1 ≤ ai ≤ 109; ai ≤ ai + 1).

The next line contains integer m (1 ≤ m ≤ 105) — the number of boxes. Each of the following m lines contains a pair of integers wi, hi (1 ≤ wi ≤ n; 1 ≤ hi ≤ 109) — the size of the i-th thrown box.

The numbers in the lines are separated by spaces.

Output

Print m integers — for each box the height, where the bottom of the box will be after landing. Print the answers for the boxes in the order, in which the boxes are given in the input.

Please, do not use the %lld specifier to read or write 64-bit integers in C++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5
1 2 3 6 6
4
1 1
3 1
1 1
4 3


Output

1
3
4
6


Input

3
1 2 3
2
1 1
3 1


Output

1
3


Input

1
1
5
1 2
1 10
1 10
1 10
1 10


Output

1
3
13
23
33

Note

The first sample are shown on the picture.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#! /Library/Frameworks/Python.framework/Versions/2.6/bin/python

n = int(raw_input())

inp = raw_input().split()
a = [int(inp[i]) for i in range(n)]

m = int(raw_input())

box_top = 0

for i in range(m):
	inp = raw_input().split()
	w = int(inp[0])
	h = int(inp[1])

	box_bottom = max(a[w-1],box_top)
	box_top = box_bottom + h

	print box_bottom
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cdimaandstaircasebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'max_stairs': params.get('max_stairs', 10),
            'max_boxes': params.get('max_boxes', 10),
            'max_height_step': params.get('max_height_step', 100),
            'max_h': params.get('max_h', 100),
        }

    def case_generator(self):
        n = random.randint(1, self.params['max_stairs'])
        
        a = []
        current = random.randint(1, 10)
        a.append(current)
        for _ in range(n-1):
            current += random.randint(0, self.params['max_height_step'])
            a.append(current)

        m = random.randint(1, self.params['max_boxes'])
        boxes = []
        for _ in range(m):
            wi = random.choice([
                random.randint(1, max(1, n//2)),
                random.randint(max(1, n//2), n),
                1,
                n
            ])
            hi = random.randint(1, self.params['max_h'])
            boxes.append((wi, hi))

        expected = []
        current_max = 0
        for w, h in boxes:
            stair_height = a[w-1]
            box_bottom = max(stair_height, current_max)
            expected.append(box_bottom)
            current_max = box_bottom + h

        return {
            "n": n,
            "a": a,
            "m": m,
            "boxes": boxes,
            "expected_outputs": expected
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        # 单独生成带有换行的描述部分
        boxes_desc = []
        for i, (w, h) in enumerate(question_case['boxes']):
            boxes_desc.append(f"第{i+1}个箱子：宽度 {w}，高度 {h}")
        boxes_str = "\n".join(boxes_desc)

        input_lines = [
            str(question_case['n']),
            ' '.join(map(str, question_case['a'])),
            str(question_case['m'])
        ] + [f"{w} {h}" for w, h in question_case['boxes']]
        input_example = '\n'.join(input_lines)

        return f"""## 楼梯箱子问题

Dima有一个包含{question_case['n']}个台阶的楼梯，台阶高度为非递减序列：{' '.join(map(str, question_case['a']))}。现在依次投掷{question_case['m']}个箱子：

{boxes_str}

**规则说明**：
1. 每个箱子会覆盖前w个台阶
2. 箱子落地高度取决于台阶高度和之前堆叠的箱子的最大值
3. 输出结果应为每个箱子底部的最终高度

请严格按照顺序输出每个结果，每个数值单独一行放在[answer]标签内。

示例格式：
[answer]
42
13
7
[/answer]

当前问题的输入数据：
{input_example}
请计算并输出结果："""  # 修复换行符问题

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        numbers = re.findall(r'\b\d+\b', answer_blocks[-1])
        try:
            return [int(num) for num in numbers]
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_outputs']
        return len(solution) == len(expected) and all(a == b for a, b in zip(solution, expected))
