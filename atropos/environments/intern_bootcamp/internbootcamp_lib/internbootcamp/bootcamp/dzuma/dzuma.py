"""# 

### 谜题描述
Genos recently installed the game Zuma on his phone. In Zuma there exists a line of n gemstones, the i-th of which has color ci. The goal of the game is to destroy all the gemstones in the line as quickly as possible.

In one second, Genos is able to choose exactly one continuous substring of colored gemstones that is a palindrome and remove it from the line. After the substring is removed, the remaining gemstones shift to form a solid line again. What is the minimum number of seconds needed to destroy the entire line?

Let us remind, that the string (or substring) is called palindrome, if it reads same backwards or forward. In our case this means the color of the first gemstone is equal to the color of the last one, the color of the second gemstone is equal to the color of the next to last and so on.

Input

The first line of input contains a single integer n (1 ≤ n ≤ 500) — the number of gemstones.

The second line contains n space-separated integers, the i-th of which is ci (1 ≤ ci ≤ n) — the color of the i-th gemstone in a line.

Output

Print a single integer — the minimum number of seconds needed to destroy the entire line.

Examples

Input

3
1 2 1


Output

1


Input

3
1 2 3


Output

3


Input

7
1 4 4 2 3 2 1


Output

2

Note

In the first sample, Genos can destroy the entire line in one second.

In the second sample, Genos can only destroy one gemstone at a time, so destroying three gemstones takes three seconds.

In the third sample, to achieve the optimal time of two seconds, destroy palindrome 4 4 first and then destroy palindrome 1 2 3 2 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=int(raw_input())
colors=map(int, raw_input().split())
#colors=range(1,n+1)

mem=[]
for _ in range(n):
    mem.append([-1]*n)

def calc( fr, to):
    if to<fr:
        return 0
    if to==fr:
        return 1
    if mem[fr][to]==-1:
        val=n;
        for k in xrange(fr, to):
            val=min(val, calc(fr, k)+calc(k+1, to))
        if colors[fr]==colors[to]:
            val=min(val, max(1,calc(fr+1, to-1)))
        mem[fr][to]=val
    
    return mem[fr][to]
    
print calc(0, n-1)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dzumabootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=15, color_range=5):
        self.min_n = min_n
        self.max_n = max_n
        self.color_range = color_range

    def case_generator(self):
        """生成任意随机序列，确保包含各类测试案例"""
        n = random.randint(self.min_n, self.max_n)
        colors = [random.randint(1, self.color_range) for _ in range(n)]
        expected = self._compute_min_steps(colors)
        return {'n': n, 'colors': colors, 'expected': expected}

    @staticmethod
    def _compute_min_steps(colors):
        """动态规划算法重构版"""
        n = len(colors)
        dp = [[0]*n for _ in range(n)]
        
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if i == j:
                    dp[i][j] = 1
                    continue
                
                # 基准情况：逐个消除
                dp[i][j] = dp[i][j-1] + 1
                
                # 处理相同颜色相邻的情况
                if colors[j] == colors[j-1]:
                    dp[i][j] = min(dp[i][j], dp[i][j-2] + 1 if j-2 >= i else 1)
                
                # 遍历所有可能的分割点
                for k in range(i, j):
                    if colors[k] == colors[j]:
                        dp[i][j] = min(dp[i][j], dp[i][k] + (dp[k+1][j-1] if k+1 <= j-1 else 0))
                
                # 处理端点相同的情况
                if colors[i] == colors[j]:
                    if j - i > 1:
                        dp[i][j] = min(dp[i][j], dp[i+1][j-1])
                    else:
                        dp[i][j] = 1
        return dp[0][n-1]

    @staticmethod
    def prompt_func(question_case) -> str:
        colors = ' '.join(map(str, question_case['colors']))
        return f"""在祖玛游戏中，你需要消除一行宝石。规则如下：
1. 每步可消除一个连续回文子串
2. 回文指正读反读相同的序列
当前宝石序列（共{question_case['n']}个）：{colors}
请输出最少需要多少秒，将答案放在[answer][/answer]中。如：[answer]2[/answer]"""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(answers[-1]) if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
