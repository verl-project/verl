"""# 

### 谜题描述
A lot of people associate Logo programming language with turtle graphics. In this case the turtle moves along the straight line and accepts commands \"T\" (\"turn around\") and \"F\" (\"move 1 unit forward\").

You are given a list of commands that will be given to the turtle. You have to change exactly n commands from the list (one command can be changed several times). How far from the starting point can the turtle move after it follows all the commands of the modified list?

Input

The first line of input contains a string commands — the original list of commands. The string commands contains between 1 and 100 characters, inclusive, and contains only characters \"T\" and \"F\".

The second line contains an integer n (1 ≤ n ≤ 50) — the number of commands you have to change in the list.

Output

Output the maximum distance from the starting point to the ending point of the turtle's path. The ending point of the turtle's path is turtle's coordinate after it follows all the commands of the modified list.

Examples

Input

FT
1


Output

2


Input

FFFTFFF
2


Output

6

Note

In the first example the best option is to change the second command (\"T\") to \"F\" — this way the turtle will cover a distance of 2 units.

In the second example you have to change two commands. One of the ways to cover maximal distance of 6 units is to change the fourth command and first or last one.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
debug = False

import sys

d = [-1, 1]

def out():
    FIN.close()
    FOUT.close()
    sys.exit()



if debug:
    FIN = open('input.txt', 'r')
    FOUT = open('output.txt', 'w')
else:
    FIN = sys.stdin
    FOUT = sys.stdout
    
s = FIN.readline()
if s[-1] == '\n':
    s = s[:-1]
n = int(FIN.readline())
fm = [[[0 if (t == 1) and(j == 0) and (i == 0) else None for t in range(2)] for j in range(n + 1)] for i in range(len(s) + 1)]
fM = [[[0 if (t == 1) and(j == 0) and (i == 0) else None for t in range(2)] for j in range(n + 1)] for i in range(len(s) + 1)]

for i in range(1, len(s) + 1):
    for j in range(n + 1):
        if s[i - 1] == 'F':
            for k in range(0, j + 1, 2):
                for t in range(2):
                    if fM[i - 1][j - k][t] is not None:
                        ff = fM[i - 1][j - k][t] + d[t]
                        if fM[i][j][t] is None or ff > fM[i][j][t]:
                            fM[i][j][t] = ff
            for k in range(1, j + 1, 2):
                for t in range(2):
                    ff = fM[i - 1][j - k][t ^ 1]
                    if (ff is not None) and (fM[i][j][t] is None or ff > fM[i][j][t]):
                        fM[i][j][t] = ff
            for k in range(0, j + 1, 2):
                for t in range(2):
                    if fm[i - 1][j - k][t] is not None:
                        ff = fm[i - 1][j - k][t] + d[t]
                        if fm[i][j][t] is None or ff < fm[i][j][t]:
                            fm[i][j][t] = ff
            for k in range(1, j + 1, 2):
                for t in range(2):
                    ff = fm[i - 1][j - k][t ^ 1]
                    if (ff is not None) and (fm[i][j][t] is None or ff < fm[i][j][t]):
                        fm[i][j][t] = ff
        else:
            for k in range(0, j + 1, 2):
                for t in range(2):
                    ff = fM[i - 1][j - k][t ^ 1]
                    if (ff is not None) and (fM[i][j][t] is None or ff > fM[i][j][t]):
                        fM[i][j][t] = ff
            for k in range(1, j + 1, 2):
                for t in range(2):
                    if fM[i - 1][j - k][t] is not None:
                        ff = fM[i - 1][j - k][t] + d[t]
                        if fM[i][j][t] is None or ff > fM[i][j][t]:
                            fM[i][j][t] = ff
            for k in range(0, j + 1, 2):
                for t in range(2):
                    ff = fm[i - 1][j - k][t ^ 1]
                    if (ff is not None) and (fm[i][j][t] is None or ff < fm[i][j][t]):
                        fm[i][j][t] = ff
            for k in range(1, j + 1, 2):
                for t in range(2):
                    if fm[i - 1][j - k][t] is not None:
                        ff = fm[i - 1][j - k][t] + d[t]
                        if fm[i][j][t] is None or ff < fm[i][j][t]:
                            fm[i][j][t] = ff
                            
if fm[len(s)][n][0] is None:
    fm[len(s)][n][0] = 0
if fm[len(s)][n][1] is None:
    fm[len(s)][n][1] = 0  
if fM[len(s)][n][0] is None:
    fM[len(s)][n][0] = 0
if fM[len(s)][n][1] is None:
    fM[len(s)][n][1] = 0  
ans = max(abs(fm[len(s)][n][0]), abs(fm[len(s)][n][1]), abs(fM[len(s)][n][0]), abs(fM[len(s)][n][1]))
FOUT.write(str(ans))
out()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def max_turtle_distance(commands: str, n: int) -> int:
    len_commands = len(commands)
    directions = [-1, 1]  # 方向映射：0-负方向，1-正方向
    
    # 初始化动态规划表
    min_dp = [[[float('inf')]*2 for _ in range(n+1)] for __ in range(len_commands+1)]
    max_dp = [[[-float('inf')]*2 for __ in range(n+1)] for ___ in range(len_commands+1)]
    
    # 初始状态：处理0个命令，0次修改，方向为正向
    min_dp[0][0][1] = 0
    max_dp[0][0][1] = 0
    
    for step in range(1, len_commands+1):
        cmd = commands[step-1]
        for changes in range(n+1):
            # 当前命令可以改变的次数（奇偶性必须满足总修改次数）
            for parity in [0, 1]:  # 0表示偶数次改变，1表示奇数次
                if changes < parity:
                    continue
                
                # 实际修改次数为k=changes - used_changes
                used_changes = changes - parity
                
                # 原始命令的效果
                if cmd == 'F':
                    movement = directions[1] if parity % 2 == 0 else 0
                    direction_change = 0
                else:
                    movement = 0
                    direction_change = 1 if parity % 2 == 0 else 0
                
                # 计算所有可能的前置状态
                for prev_dir in [0, 1]:
                    if min_dp[step-1][used_changes][prev_dir] == float('inf'):
                        continue
                    
                    new_dir = prev_dir ^ direction_change
                    new_pos = min_dp[step-1][used_changes][prev_dir] + movement * directions[prev_dir]
                    
                    if new_pos < min_dp[step][changes][new_dir]:
                        min_dp[step][changes][new_dir] = new_pos
                    if new_pos > max_dp[step][changes][new_dir]:
                        max_dp[step][changes][new_dir] = new_pos

    # 收集所有可能的最终状态
    candidates = []
    for d in [0, 1]:
        candidates.append(abs(min_dp[len_commands][n][d]))
        candidates.append(abs(max_dp[len_commands][n][d]))
    
    return max(candidates) if candidates else 0

class Clogoturtlebootcamp(Basebootcamp):
    def __init__(self, min_commands_length=1, max_commands_length=100, min_n=1, max_n=50):
        self.min_commands_length = min_commands_length
        self.max_commands_length = max_commands_length
        self.min_n = min(min_n, 50)  # 强制限制题目参数范围
        self.max_n = min(max_n, 50)
    
    def case_generator(self):
        # 生成符合题目要求的合法测试用例
        length = random.randint(self.min_commands_length, self.max_commands_length)
        commands = ''.join(random.choices(['F', 'T'], k=length))
        n = random.randint(self.min_n, self.max_n)  # 保证n在有效范围内
        return {'commands': commands, 'n': n}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        commands = question_case['commands']
        n = question_case['n']
        return f"""请解决以下乌龟移动优化问题：

问题说明：
乌龟在直线上移动，初始方向为正方向。命令序列由以下字符组成：
F - 前进1单位（保持当前方向）
T - 反转移动方向
你需要恰好修改{n}个命令（每次修改可将任意字符变为另一种），求最终可能达到的距起点最大绝对距离。

输入参数：
命令序列：{commands}
必须修改次数：{n}

请将最终答案（整数）置于[answer]标签内，例如：[answer]42[/answer]
确保：
1. 只保留最终数值答案
2. 严格遵循答案格式要求"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            ground_truth = max_turtle_distance(
                identity['commands'], 
                identity['n']
            )
            return solution == ground_truth
        except:
            return False
