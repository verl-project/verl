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
from sys import stdin, stdout
from collections import Counter, defaultdict
from itertools import permutations, combinations
raw_input = stdin.readline
pr = stdout.write


def in_num():
    return int(raw_input())


def in_arr():
    return map(int,raw_input().split())


def pr_num(n):
    stdout.write(str(n)+'\n')


def pr_arr(arr):
    pr(' '.join(map(str,arr))+'\n')

# fast read function for total integer input

def inp():
    # this function returns whole input of
    # space/line seperated integers
    # Use Ctrl+D to flush stdin.
    return map(int,stdin.read().split())

range = xrange # not for python 3.0+

# main code

n=in_num()
l=in_arr()
dp=[[1 for i in range(n)] for j in range(n)]
for ln in range(1,n):
    for i in range(n-ln):
        dp[i][i+ln]=min(dp[i][j]+dp[j+1][i+ln] for j in range(i,i+ln))
        
        if l[i]==l[i+ln]:
            dp[i][i+ln]=min(dp[i][i+ln],dp[i+1][i+ln-1])
pr_num(dp[0][-1])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Bzumabootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_n = params.get('min_n', 1)
        self.max_n = params.get('max_n', 10)
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        colors = self._generate_valid_colors(n)
        min_steps = self._calculate_min_steps(colors)
        return {
            'n': n,
            'colors': colors,
            'min_steps': min_steps
        }
    
    def _generate_valid_colors(self, n):
        """生成至少有一个回文子串的合法颜色序列"""
        colors = []
        for _ in range(n):
            # 50%概率延续当前颜色形成回文
            if colors and random.random() < 0.3:
                colors.append(random.choice(colors))
            else:
                colors.append(random.randint(1, n))
        return colors
    
    def _calculate_min_steps(self, colors):
        n = len(colors)
        if n == 0:
            return 0
        
        dp = [[0]*n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        
        for ln in range(1, n):
            for i in range(n - ln):
                j = i + ln
                dp[i][j] = min(dp[i][k] + dp[k+1][j] for k in range(i, j))
                
                if colors[i] == colors[j]:
                    if ln == 1:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i][j], dp[i+1][j-1])
        
        return dp[0][n-1]

    @staticmethod
    def prompt_func(question_case):
        colors = question_case['colors']
        example = """
示例1：
输入：3
      1 2 1
输出：1（直接消除整个回文）

示例2：
输入：3
      1 2 3 
输出：3（每次只能消除一个）"""
        return f"""祖玛游戏问题：给定{question_case['n']}个宝石组成的序列{colors}，
每次可以消除一个连续回文子串，求完全消除需要的最少次数。

规则说明：
1. 回文子串长度至少为1
2. 消除操作后剩余宝石会自动合并
3. 需保证策略最优性

请给出准确的最小操作次数，答案用[answer]标签包裹，如：[answer]2[/answer]。
{example}"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        numbers = re.findall(r'-?\d+', last_match)
        return int(numbers[-1]) if numbers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, int) or solution < 1:
            return False
        return solution == identity['min_steps']
