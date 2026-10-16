"""# 

### 谜题描述
Santa Claus has Robot which lives on the infinite grid and can move along its lines. He can also, having a sequence of m points p1, p2, ..., pm with integer coordinates, do the following: denote its initial location by p0. First, the robot will move from p0 to p1 along one of the shortest paths between them (please notice that since the robot moves only along the grid lines, there can be several shortest paths). Then, after it reaches p1, it'll move to p2, again, choosing one of the shortest ways, then to p3, and so on, until he has visited all points in the given order. Some of the points in the sequence may coincide, in that case Robot will visit that point several times according to the sequence order.

While Santa was away, someone gave a sequence of points to Robot. This sequence is now lost, but Robot saved the protocol of its unit movements. Please, find the minimum possible length of the sequence.

Input

The first line of input contains the only positive integer n (1 ≤ n ≤ 2·105) which equals the number of unit segments the robot traveled. The second line contains the movements protocol, which consists of n letters, each being equal either L, or R, or U, or D. k-th letter stands for the direction which Robot traveled the k-th unit segment in: L means that it moved to the left, R — to the right, U — to the top and D — to the bottom. Have a look at the illustrations for better explanation.

Output

The only line of input should contain the minimum possible length of the sequence.

Examples

Input

4
RURD


Output

2


Input

6
RRULDD


Output

2


Input

26
RRRULURURUULULLLDLDDRDRDLD


Output

7


Input

3
RLL


Output

2


Input

4
LRLR


Output

4

Note

The illustrations to the first three tests are given below.

<image> <image> <image>

The last example illustrates that each point in the sequence should be counted as many times as it is presented in the sequence.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin

n, s, ans, dis, rev = int(input()), stdin.readline().strip(), 1, set(), {'R': 'L', 'L': 'R', 'U': 'D', 'D': 'U'}
for i in s:
    if rev[i] in dis:
        ans += 1
        dis = {i}
    else:
        dis.add(i)
print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Csantaclausandrobotbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=20):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        directions = ['L', 'R', 'U', 'D']
        movements = ''.join(random.choices(directions, k=n))
        correct_answer = self.calculate_answer(movements)
        return {
            'n': n,
            'movements': movements,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def calculate_answer(s):
        ans = 1
        dis = set()
        rev = {'R': 'L', 'L': 'R', 'U': 'D', 'D': 'U'}
        for c in s:
            if rev[c] in dis:
                ans += 1
                dis = {c}
            else:
                dis.add(c)
        return ans
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = question_case['movements']
        return f"""你是Santa的助手，需要根据机器人的移动协议确定可能的最短点序列长度。机器人移动规则如下：

1. 机器人按点序列p₁,p₂,...,pₘ移动，每次必须走两点间的最短路径
2. 移动协议中的每个字符代表一个单位移动方向（L/R/U/D）
3. 当移动方向的相反方向已在当前允许方向集中时，必须开始新的阶段并增加点序列长度

输入：
- 第一行是移动单元数n={n}
- 第二行是移动协议：{s}

请计算最小可能的点序列长度，并将答案放在[answer]和[/answer]标记之间，如：[answer]答案[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['correct_answer']
        except (ValueError, KeyError, TypeError):
            return False
