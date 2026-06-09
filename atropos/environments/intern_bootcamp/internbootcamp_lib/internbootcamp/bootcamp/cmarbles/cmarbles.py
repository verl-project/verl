"""# 

### 谜题描述
In the spirit of the holidays, Saitama has given Genos two grid paths of length n (a weird gift even by Saitama's standards). A grid path is an ordered sequence of neighbouring squares in an infinite grid. Two squares are neighbouring if they share a side.

One example of a grid path is (0, 0) → (0, 1) → (0, 2) → (1, 2) → (1, 1) → (0, 1) → ( - 1, 1). Note that squares in this sequence might be repeated, i.e. path has self intersections.

Movement within a grid path is restricted to adjacent squares within the sequence. That is, from the i-th square, one can only move to the (i - 1)-th or (i + 1)-th squares of this path. Note that there is only a single valid move from the first and last squares of a grid path. Also note, that even if there is some j-th square of the path that coincides with the i-th square, only moves to (i - 1)-th and (i + 1)-th squares are available. For example, from the second square in the above sequence, one can only move to either the first or third squares.

To ensure that movement is not ambiguous, the two grid paths will not have an alternating sequence of three squares. For example, a contiguous subsequence (0, 0) → (0, 1) → (0, 0) cannot occur in a valid grid path.

One marble is placed on the first square of each grid path. Genos wants to get both marbles to the last square of each grid path. However, there is a catch. Whenever he moves one marble, the other marble will copy its movement if possible. For instance, if one marble moves east, then the other marble will try and move east as well. By try, we mean if moving east is a valid move, then the marble will move east.

Moving north increases the second coordinate by 1, while moving south decreases it by 1. Similarly, moving east increases first coordinate by 1, while moving west decreases it.

Given these two valid grid paths, Genos wants to know if it is possible to move both marbles to the ends of their respective paths. That is, if it is possible to move the marbles such that both marbles rest on the last square of their respective paths.

Input

The first line of the input contains a single integer n (2 ≤ n ≤ 1 000 000) — the length of the paths.

The second line of the input contains a string consisting of n - 1 characters (each of which is either 'N', 'E', 'S', or 'W') — the first grid path. The characters can be thought of as the sequence of moves needed to traverse the grid path. For example, the example path in the problem statement can be expressed by the string \"NNESWW\".

The third line of the input contains a string of n - 1 characters (each of which is either 'N', 'E', 'S', or 'W') — the second grid path.

Output

Print \"YES\" (without quotes) if it is possible for both marbles to be at the end position at the same time. Print \"NO\" (without quotes) otherwise. In both cases, the answer is case-insensitive.

Examples

Input

7
NNESWW
SWSWSW


Output

YES


Input

3
NN
SS


Output

NO

Note

In the first sample, the first grid path is the one described in the statement. Moreover, the following sequence of moves will get both marbles to the end: NNESWWSWSW.

In the second sample, no sequence of moves can get both marbles to the end.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os

def main_entry():
    c2i = {'N':0, 'S':1, 'W':2, 'E':3}
    while True:
        try:
            n = int(raw_input())
        except ValueError:
            continue
        except EOFError:
            break
        a = str(raw_input())[::-1]
        b = str(raw_input())
        c = ''
        for k in a:
            c += str(c2i[k]^1)
        c += '#'
        for k in b:
            c += str(c2i[k])
        n = len(c)
        f = [-1 for i in xrange(n)]
        for i in range(1, n):
            k = f[i-1]
            while k>=0 and c[i]!=c[k+1]:
                k = f[k]
            if c[i]==c[k+1]:
                f[i] = k+1
        if f[n-1]==-1:
            print 'YES'
        else:
            print 'NO'

if __name__ == '__main__':
    main_entry()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cmarblesbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=20):
        self.n_min = n_min
        self.n_max = n_max
        self.delta = {
            'N': (0, 1),
            'S': (0, -1),
            'E': (1, 0),
            'W': (-1, 0)
        }
    
    def generate_valid_path(self, length):
        directions = ['N', 'E', 'S', 'W']
        pos = [(0, 0)]
        path = []
        for _ in range(length):
            available = []
            current_pos = pos[-1]
            if len(pos) >= 2:
                prev_prev_pos = pos[-2]
                for d in directions:
                    dx, dy = self.delta[d]
                    new_x = current_pos[0] + dx
                    new_y = current_pos[1] + dy
                    new_pos = (new_x, new_y)
                    if new_pos != prev_prev_pos:
                        available.append(d)
            else:
                available = directions.copy()
            if not available:
                return None
            d = random.choice(available)
            dx, dy = self.delta[d]
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            pos.append(new_pos)
            path.append(d)
        return ''.join(path)
    
    def case_generator(self):
        while True:
            n = random.randint(self.n_min, self.n_max)
            path1 = self.generate_valid_path(n-1)
            path2 = self.generate_valid_path(n-1)
            if path1 is not None and path2 is not None:
                return {
                    'n': n,
                    'path1': path1,
                    'path2': path2
                }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        path1 = question_case['path1']
        path2 = question_case['path2']
        prompt = f"""你是Genos，需要帮助判断两个弹珠是否可以同时移动到各自的路径终点。下面是问题的详细说明：

**背景与规则：**

- 两个弹珠分别位于两个不同的网格路径的起点。每个路径的长度为n，由一系列移动方向组成，每个移动方向是N（北）、E（东）、S（南）、W（西）中的一个字符。
- 每次移动时，你必须选择移动其中一个弹珠。另一个弹珠会自动尝试复制相同的移动，但如果无法移动（即该方向无法从当前位置移动到下一个位置），则不会移动。
- 移动只能在路径的相邻位置上进行。也就是说，弹珠只能沿路径的步进顺序移动，不能跳过或回退到之前的位置，除了自动复制移动的情况。
- 输入的路径保证不会出现三个连续的步骤导致坐标交替出现的情况（例如A → B → A的结构）。

**输入：**

- 第一行是整数n（2 ≤ n ≤ 1,000,000），表示路径的长度。
- 第二行是第一个路径的移动序列，由n-1个字符组成。
- 第三行是第二个路径的移动序列，同样有n-1个字符。

**任务：**

判断是否存在一种移动顺序，使得两个弹珠最终同时停留在各自的终点。如果是，输出“YES”；否则，输出“NO”。

**当前问题实例：**

n = {n}

第一个路径的移动序列：{path1}

第二个路径的移动序列：{path2}

**答案格式要求：**

请将最终答案用[answer]标签包裹，例如：[answer]YES[/answer]或[answer]NO[/answer]。确保只包含一个答案，并且是最后一次判断的结果。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        path1 = identity['path1']
        path2 = identity['path2']
        
        c2i = {'N':0, 'S':1, 'W':2, 'E':3}
        a = path1[::-1]
        c = []
        for k in a:
            c.append(str(c2i[k] ^ 1))
        c.append('#')
        for k in path2:
            c.append(str(c2i[k]))
        c_str = ''.join(c)
        
        n_total = len(c_str)
        f = [-1] * n_total
        for i in range(1, n_total):
            k = f[i-1]
            while k >= 0 and c_str[i] != c_str[k+1]:
                k = f[k]
            if c_str[i] == c_str[k+1]:
                f[i] = k + 1
            else:
                f[i] = -1
        
        correct_answer = 'YES' if f[-1] == -1 else 'NO'
        return solution.strip().upper() == correct_answer.upper()
