"""# 

### 谜题描述
The football season has just ended in Berland. According to the rules of Berland football, each match is played between two teams. The result of each match is either a draw, or a victory of one of the playing teams. If a team wins the match, it gets w points, and the opposing team gets 0 points. If the game results in a draw, both teams get d points.

The manager of the Berland capital team wants to summarize the results of the season, but, unfortunately, all information about the results of each match is lost. The manager only knows that the team has played n games and got p points for them.

You have to determine three integers x, y and z — the number of wins, draws and loses of the team. If there are multiple answers, print any of them. If there is no suitable triple (x, y, z), report about it.

Input

The first line contains four integers n, p, w and d (1 ≤ n ≤ 10^{12}, 0 ≤ p ≤ 10^{17}, 1 ≤ d < w ≤ 10^{5}) — the number of games, the number of points the team got, the number of points awarded for winning a match, and the number of points awarded for a draw, respectively. Note that w > d, so the number of points awarded for winning is strictly greater than the number of points awarded for draw.

Output

If there is no answer, print -1.

Otherwise print three non-negative integers x, y and z — the number of wins, draws and losses of the team. If there are multiple possible triples (x, y, z), print any of them. The numbers should meet the following conditions: 

  * x ⋅ w + y ⋅ d = p, 
  * x + y + z = n. 

Examples

Input


30 60 3 1


Output


17 9 4


Input


10 51 5 4


Output


-1


Input


20 0 15 5


Output


0 0 20

Note

One of the possible answers in the first example — 17 wins, 9 draws and 4 losses. Then the team got 17 ⋅ 3 + 9 ⋅ 1 = 60 points in 17 + 9 + 4 = 30 games.

In the second example the maximum possible score is 10 ⋅ 5 = 50. Since p = 51, there is no answer.

In the third example the team got 0 points, so all 20 games were lost.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from fractions import gcd

def modinv(x, n):
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = n, x
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    
    if old_r != 1: return -1 
    return old_s % n

n, p, w, d = map(int, raw_input().strip().split())

g = gcd(w, d)
if p % g != 0:
    print -1
    exit()

r = modinv(d / g, (w - d) / g)
r *= (p / g)

if r < 0:
    print -1
    exit()

q = (p - w * r + (w * (w - d) / g) - 1) / (w * (w - d) / g)
N = q * ((w - d) / g) + r

if N > n:
    print -1
    exit()

x = (p - N * d) / (w - d)
if (x < 0):
    print -1
    exit()

y = N - x
z = n - N

try:
    assert (w * x + d * y == p and x >= 0 and y >= 0 and z >= 0)
except:
    print -1
    exit()
    
print x, y, z
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import math
from math import gcd
from bootcamp import Basebootcamp

class Cthefootballseasonbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'w_min': params.get('w_min', 2),
            'w_max': params.get('w_max', 10**5),
            'd_min': params.get('d_min', 1),
            'd_max': params.get('d_max', None),
            'valid_case_ratio': params.get('valid_case_ratio', 0.5)
        }

    def case_generator(self):
        if random.random() < self.params['valid_case_ratio']:
            return self._generate_valid_case()
        return self._generate_robust_invalid_case()

    def _generate_valid_case(self):
        w = random.randint(self.params['w_min'], self.params['w_max'])
        d_max = min(w-1, self.params['d_max'] or w-1)
        d = random.randint(self.params['d_min'], d_max)

        max_games = 10**12
        x = random.randint(0, max_games)
        remaining = random.randint(0, max_games - x)
        y = random.randint(0, remaining)
        z = max_games - x - y

        n = x + y + z
        p = x * w + y * d
        return {'n': min(n, 10**12), 'p': p, 'w': w, 'd': d}

    def _generate_robust_invalid_case(self):
        w = random.randint(self.params['w_min'], self.params['w_max'])
        d_max = min(w-1, self.params['d_max'] or w-1)
        d = random.randint(self.params['d_min'], d_max)
        n = random.randint(1, 10**12)
        g = gcd(w, d)
        max_points = w * n
        
        # 生成无效p的候选策略
        invalid_candidates = []
        
        # 策略1：p超过最大可能值
        if max_points < 10**17:
            invalid_candidates.append(random.randint(max_points + 1, max_points + 1000))
        
        # 策略2：p不满足模条件（仅当gcd>1时有效）
        if g > 1:
            for _ in range(100):  # 有限次尝试
                p = random.randint(0, max_points)
                if p % g != 0:
                    invalid_candidates.append(p)
                    break
        
        # 策略3：代数无解的情况（无论模条件）
        if not invalid_candidates:
            p = random.randint(0, min(max_points, 10**17))
            while self._has_solution({'n':n, 'p':p, 'w':w, 'd':d}):
                p = random.randint(0, min(max_points, 10**17))
            invalid_candidates.append(p)
        
        return {
            'n': n,
            'p': random.choice(invalid_candidates),
            'w': w,
            'd': d
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""Determine football results (x wins, y draws, z losses) where:
- Total games: {question_case['n']}
- Points obtained: {question_case['p']}
- Win points: {question_case['w']}
- Draw points: {question_case['d']}
Answer format: [answer]x y z[/answer] or [answer]-1[/answer] if impossible."""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers:
            return None
        
        last_answer = answers[-1].strip()
        if last_answer == '-1':
            return [-1]
        
        try:
            nums = list(map(int, re.findall(r'-?\d+', last_answer)))
            if len(nums) == 3 and all(n >=0 for n in nums):
                return tuple(nums)
        except:
            pass
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == [-1]:
            return not cls._has_solution(identity)
        
        try:
            x, y, z = solution
            return (x >= 0 and y >= 0 and z >= 0 and
                    x + y + z == identity['n'] and
                    x*identity['w'] + y*identity['d'] == identity['p'])
        except:
            return False

    @classmethod
    def _has_solution(cls, case):
        # 实现参考文献解法的数学验证逻辑
        n, p, w, d = case.values()
        g = gcd(w, d)
        
        if p % g != 0 or p > w * n:
            return False
        
        m = (w - d) // g
        if m == 0:
            return p == 0
        
        try:
            inv = pow(d//g, -1, m)
            y0 = (p//g * inv) % m
            x = (p//g - (d//g)*y0) // (w//g - d//g)
            return x >= 0 and y0 >=0 and (x + y0) <= n
        except:
            return False
