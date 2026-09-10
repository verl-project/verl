"""# 

### 谜题描述
There are n games in a football tournament. Three teams are participating in it. Currently k games had already been played. 

You are an avid football fan, but recently you missed the whole k games. Fortunately, you remember a guess of your friend for these k games. Your friend did not tell exact number of wins of each team, instead he thought that absolute difference between number of wins of first and second team will be d1 and that of between second and third team will be d2.

You don't want any of team win the tournament, that is each team should have the same number of wins after n games. That's why you want to know: does there exist a valid tournament satisfying the friend's guess such that no team will win this tournament?

Note that outcome of a match can not be a draw, it has to be either win or loss.

Input

The first line of the input contains a single integer corresponding to number of test cases t (1 ≤ t ≤ 105).

Each of the next t lines will contain four space-separated integers n, k, d1, d2 (1 ≤ n ≤ 1012; 0 ≤ k ≤ n; 0 ≤ d1, d2 ≤ k) — data for the current test case.

Output

For each test case, output a single line containing either \"yes\" if it is possible to have no winner of tournament, or \"no\" otherwise (without quotes).

Examples

Input

5
3 0 0 0
3 3 0 0
6 4 1 0
6 3 3 0
3 3 3 2


Output

yes
yes
yes
no
no

Note

Sample 1. There has not been any match up to now (k = 0, d1 = 0, d2 = 0). If there will be three matches (1-2, 2-3, 3-1) and each team wins once, then at the end each team will have 1 win.

Sample 2. You missed all the games (k = 3). As d1 = 0 and d2 = 0, and there is a way to play three games with no winner of tournament (described in the previous sample), the answer is \"yes\".

Sample 3. You missed 4 matches, and d1 = 1, d2 = 0. These four matches can be: 1-2 (win 2), 1-3 (win 3), 1-2 (win 1), 1-3 (win 1). Currently the first team has 2 wins, the second team has 1 win, the third team has 1 win. Two remaining matches can be: 1-2 (win 2), 1-3 (win 3). In the end all the teams have equal number of wins (2 wins).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def get(k, d1, d2):
    x = (k + (2 * d1) + d2)
    if x < 0 or x % 3 != 0:
        return -1, -1, -1
    x /= 3
    y = (k + d1 + d2) - (2 * x)
    if y < 0:
        return -1, -1, -1
    z = (x - d1 - d2)
    if z < 0:
        return -1, -1, -1
    return x, y, z


def solve(n, k, d1, d2):
    signs = [
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1)
    ]
    for (s1, s2) in signs:
        x, y, z = get(k, s1 * d1, s2 * d2)
        if x != -1:
            remaining = (n - k)
            remaining += (x + y + z)
            if remaining % 3 == 0:
                remaining /= 3
                if remaining >= 0 and remaining >= x and remaining >= y and remaining >= z:
                    return True
    return False

[t] = map(int, raw_input('').split(' '))
for i in xrange(t):
    nums = map(int, raw_input('').split(' '))
    if solve(*nums):
        print 'yes'
    else:
        print 'no'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cpredictoutcomeofthegamebootcamp(Basebootcamp):
    def __init__(self, min_m=1, max_m=10**4):
        """
        参数控制最终胜利数m的范围（总比赛数n=3m）
        """
        super().__init__()
        self.min_m = max(1, min_m)
        self.max_m = max_m
    
    def case_generator(self):
        # 生成必然有解的案例（yes案例）
        m = random.randint(self.min_m, self.max_m)
        n = 3 * m
        
        # 在[0, m]范围内生成x,y,z，总和不超过k_max（最大不超过n）
        # k可以在0到n之间，但需要满足x+y+z=k且x,y,z <=m
        x = random.randint(0, m)
        y = random.randint(0, m)
        z = random.randint(0, m)
        k = x + y + z  # 确保k <=3m =n
        
        # 避免k超过n的情况
        if k > n:
            k = n
            x = min(x, m)
            y = min(y, m)
            z = k -x -y
            z = max(0, min(z, m))
        
        d1 = abs(x - y)
        d2 = abs(y - z)
        
        # 生成包含有效解的案例
        return {
            'n': n,
            'k': k,
            'd1': d1,
            'd2': d2,
            '_expected': 'yes'  # 标记预期答案
        }
    
    @staticmethod
    def prompt_func(question_case):
        case = question_case
        problem = (
            "作为足球锦标赛观察员，判断是否存在以下条件满足的情况：\n"
            f"- 总比赛数：{case['n']}\n"
            f"- 已进行比赛：{case['k']}\n"
            f"- 队伍1与2胜利差：{case['d1']}\n"
            f"- 队伍2与3胜利差：{case['d2']}\n"
            "规则：所有比赛必须分出胜负，最终三队胜利数完全相同。\n"
            "答案用[answer]标签包裹，例如：[answer]yes[/answer]"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](yes|no)\[/answer\]', output, re.IGNORECASE)
        return matches[-1].lower() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('_expected', None)
        if expected:
            return solution == expected
        else:
            n, k, d1, d2 = identity['n'], identity['k'], identity['d1'], identity['d2']
            return solution == ('yes' if cls.solve(n, k, d1, d2) else 'no')
    
    @staticmethod
    def solve(n, k, d1, d2):
        if n % 3 != 0:
            return False
        m = n // 3
        signs = [(1,1), (1,-1), (-1,1), (-1,-1)]
        for s1, s2 in signs:
            adjusted_d1 = s1 * d1
            adjusted_d2 = s2 * d2
            x = (k + 2*adjusted_d1 + adjusted_d2)
            if x < 0 or x % 3 != 0:
                continue
            x = x // 3
            y = (k + adjusted_d1 + adjusted_d2) - 2*x
            if y < 0 or (y + x) < adjusted_d1:
                continue
            z = x - adjusted_d1 - adjusted_d2
            if z < 0:
                continue
            if (x + y + z) != k:
                continue
            if m >= x and m >= y and m >= z:
                return True
        return False
