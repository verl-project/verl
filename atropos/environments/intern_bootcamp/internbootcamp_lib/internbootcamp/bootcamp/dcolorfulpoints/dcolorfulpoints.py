"""# 

### 谜题描述
You are given a set of points on a straight line. Each point has a color assigned to it. For point a, its neighbors are the points which don't have any other points between them and a. Each point has at most two neighbors - one from the left and one from the right.

You perform a sequence of operations on this set of points. In one operation, you delete all points which have a neighbor point of a different color than the point itself. Points are deleted simultaneously, i.e. first you decide which points have to be deleted and then delete them. After that you can perform the next operation etc. If an operation would not delete any points, you can't perform it.

How many operations will you need to perform until the next operation does not have any points to delete?

Input

Input contains a single string of lowercase English letters 'a'-'z'. The letters give the points' colors in the order in which they are arranged on the line: the first letter gives the color of the leftmost point, the second gives the color of the second point from the left etc.

The number of the points is between 1 and 106.

Output

Output one line containing an integer - the number of operations which can be performed on the given set of points until there are no more points to delete.

Examples

Input

aabb


Output

2


Input

aabcaa


Output

1

Note

In the first test case, the first operation will delete two middle points and leave points \"ab\", which will be deleted with the second operation. There will be no points left to apply the third operation to.

In the second test case, the first operation will delete the four points in the middle, leaving points \"aa\". None of them have neighbors of other colors, so the second operation can't be applied.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s=raw_input()
a=[[s[0],1]]
for i in s[1:]:
    if(a[-1][0]==i):
        a[-1][1]+=1
    else:
        a.append([i,1])
turns=0
while((len(a)>1)):
    turns+=1
    temp=[]
    if(a[0][1]>1):
        temp.append([a[0][0],a[0][1]-1])
    for i in a[1:-1]:
        if(i[1]>2):
            temp.append([i[0],i[1]-2])
    if(a[-1][1]>1):
        temp.append([a[-1][0],a[-1][1]-1])
    if(len(temp)<2):
        break
    a=[temp[0],]
    for i in temp[1:]:
        if(i[0]!=a[-1][0]):
            a.append(i)
        else:
            a[-1][1]+=i[1]
print(turns)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from collections import deque
from bootcamp import Basebootcamp

class Dcolorfulpointsbootcamp(Basebootcamp):
    def __init__(self, min_blocks=3, max_blocks=10, max_block_length=5):
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.max_block_length = max_block_length
    
    def case_generator(self):
        # 生成包含多颜色块的复杂结构
        num_blocks = random.randint(self.min_blocks, self.max_blocks)
        blocks = []
        prev_color = None
        for _ in range(num_blocks):
            while True:
                color = random.choice(string.ascii_lowercase)
                if color != prev_color:
                    prev_color = color
                    break
            length = random.randint(1, self.max_block_length)
            blocks.append((color, length))
        input_str = ''.join(c * l for c, l in blocks)
        return {
            'input': input_str,
            'blocks': blocks.copy(),  # 保存原始块结构用于验证
            'output': self._calculate_operations(blocks)
        }

    def _calculate_operations(self, initial_blocks):
        blocks = deque(initial_blocks)
        turns = 0
        
        while True:
            remove_mask = [False] * len(blocks)
            
            # 标记需要删除的块
            for i in range(len(blocks)):
                left = i > 0 and blocks[i-1][0] != blocks[i][0]
                right = i < len(blocks)-1 and blocks[i+1][0] != blocks[i][0]
                if left or right:
                    remove_mask[i] = True

            # 检查是否还有可删除元素
            if not any(remove_mask):
                return turns

            # 执行删除并重新构建块结构
            new_blocks = []
            for i, (c, l) in enumerate(blocks):
                if not remove_mask[i]:
                    new_blocks.append((c, l))
                else:
                    if l > 1:
                        new_blocks.append((c, l-1))

            # 合并相邻同色块
            merged = []
            for c, l in new_blocks:
                if merged and merged[-1][0] == c:
                    merged[-1] = (c, merged[-1][1] + l)
                else:
                    merged.append((c, l))
            
            # 更新块结构
            blocks = deque(merged)
            turns += 1
            
            # 终止条件：只剩一个块且不需要删除
            if len(blocks) == 1 and not remove_mask[0]:
                return turns

    @staticmethod
    def prompt_func(question_case) -> str:
        input_str = question_case['input']
        return f"""给定直线上颜色点的排列顺序，计算完整删除操作次数。字符串表示为：{input_str}

操作规则：
1. 每次删除所有具有异色邻居的点（左右相邻颜色不同）
2. 删除是同时进行的
3. 重复直到无法删除为止

答案格式：[answer]整数[/answer]，如[answer]2[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)\[\/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['output']
