"""# 

### 谜题描述
Ziota found a video game called \"Monster Invaders\".

Similar to every other shooting RPG game, \"Monster Invaders\" involves killing monsters and bosses with guns.

For the sake of simplicity, we only consider two different types of monsters and three different types of guns.

Namely, the two types of monsters are: 

  * a normal monster with 1 hp. 
  * a boss with 2 hp. 



And the three types of guns are: 

  * Pistol, deals 1 hp in damage to one monster, r_1 reloading time 
  * Laser gun, deals 1 hp in damage to all the monsters in the current level (including the boss), r_2 reloading time 
  * AWP, instantly kills any monster, r_3 reloading time 



The guns are initially not loaded, and the Ziota can only reload 1 gun at a time.

The levels of the game can be considered as an array a_1, a_2, …, a_n, in which the i-th stage has a_i normal monsters and 1 boss. Due to the nature of the game, Ziota cannot use the Pistol (the first type of gun) or AWP (the third type of gun) to shoot the boss before killing all of the a_i normal monsters.

If Ziota damages the boss but does not kill it immediately, he is forced to move out of the current level to an arbitrary adjacent level (adjacent levels of level i (1 < i < n) are levels i - 1 and i + 1, the only adjacent level of level 1 is level 2, the only adjacent level of level n is level n - 1). Ziota can also choose to move to an adjacent level at any time. Each move between adjacent levels are managed by portals with d teleportation time.

In order not to disrupt the space-time continuum within the game, it is strictly forbidden to reload or shoot monsters during teleportation. 

Ziota starts the game at level 1. The objective of the game is rather simple, to kill all the bosses in all the levels. He is curious about the minimum time to finish the game (assuming it takes no time to shoot the monsters with a loaded gun and Ziota has infinite ammo on all the three guns). Please help him find this value.

Input

The first line of the input contains five integers separated by single spaces: n (2 ≤ n ≤ 10^6) — the number of stages, r_1, r_2, r_3 (1 ≤ r_1 ≤ r_2 ≤ r_3 ≤ 10^9) — the reload time of the three guns respectively, d (1 ≤ d ≤ 10^9) — the time of moving between adjacent levels.

The second line of the input contains n integers separated by single spaces a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^6, 1 ≤ i ≤ n).

Output

Print one integer, the minimum time to finish the game.

Examples

Input


4 1 3 4 3
3 2 5 1


Output


34

Input


4 2 4 4 1
4 5 1 2


Output


31

Note

In the first test case, the optimal strategy is:

  * Use the pistol to kill three normal monsters and AWP to kill the boss (Total time 1⋅3+4=7) 
  * Move to stage two (Total time 7+3=10) 
  * Use the pistol twice and AWP to kill the boss (Total time 10+1⋅2+4=16) 
  * Move to stage three (Total time 16+3=19) 
  * Use the laser gun and forced to move to either stage four or two, here we move to stage four (Total time 19+3+3=25) 
  * Use the pistol once, use AWP to kill the boss (Total time 25+1⋅1+4=30) 
  * Move back to stage three (Total time 30+3=33) 
  * Kill the boss at stage three with the pistol (Total time 33+1=34) 



Note that here, we do not finish at level n, but when all the bosses are killed.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#   Author: yumtam
#   Created at: 2020-09-08 13:03

from __future__ import division, print_function
_interactive = False

def main():
    n, r1, r2, r3, d = input_as_list()
    ar = input_as_list()

    r1, r2, r3, d = map(float, (r1, r2, r3, d))
    dp = array_of(lambda: INF, n, 3)

    dp[0][0] = r1*ar[0] + r3            #1
    dp[0][1] = min(r2+r1, r1*(ar[0]+2)) #2F
    dp[0][2] = INF                      #2S

    for i in range(n-1):
        c1 = r1*ar[i+1] + r3
        c2 = min(r2+r1, r1*(ar[i+1]+2))
        #1->1
        dp[i+1][0] = min(dp[i+1][0], dp[i][0]+c1+d)
        #1->2F
        penalty = (2*d if i==n-2 else 0)
        dp[i+1][1] = min(dp[i+1][1], dp[i][0]+c2+d+penalty)
        #2F->1, if this is last move, d is discounted
        discount = (d if i==n-2 else 0)
        dp[i+1][0] = min(dp[i+1][0], dp[i][1]+c1+3*d-discount)
        #2F->2S
        dp[i+1][2] = min(dp[i+1][2], dp[i][1]+c2+3*d)
        #2S->1
        dp[i+1][0] = min(dp[i+1][0], dp[i][2]+c1+d)
        #2S->2F
        penalty = (2*d if i==n-2 else 0)
        dp[i+1][1] = min(dp[i+1][1], dp[i][2]+c2+d+penalty)

    debug_print(dp)
    print(int(min(dp[-1])))


# Constants
INF = float('inf')
MOD = 10**9+7

# Python3 equivalent names
import os, sys, itertools
if sys.version_info[0] < 3:
    input = raw_input
    range = xrange

    filter = itertools.ifilter
    map = itertools.imap
    zip = itertools.izip

# print-flush in interactive problems
if _interactive:
    flush = sys.stdout.flush
    def printf(*args, **kwargs):
        print(*args, **kwargs)
        flush()

# Debug print, only works on local machine
LOCAL = \"LOCAL_\" in os.environ
debug_print = (print) if LOCAL else (lambda *x, **y: None)

# Fast IO
if (not LOCAL) and (not _interactive):
    from io import BytesIO
    from atexit import register
    sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
    sys.stdout = BytesIO()
    register(lambda: os.write(1, sys.stdout.getvalue()))
    input = lambda: sys.stdin.readline().rstrip('\r\n')

# Some utility functions(Input, N-dimensional lists, ...)
def input_as_list():
    return [int(x) for x in input().split()]

def input_with_offset(o):
    return [int(x)+o for x in input().split()]

def input_as_matrix(n, m):
    return [input_as_list() for _ in range(n)]

def array_of(f, *dim):
    return [array_of(f, *dim[1:]) for _ in range(dim[0])] if dim else f()

# Start of external code templates...
# End of external code templates.

main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

class Cmonsterinvadersbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_n = params.get('min_n', 2)
        self.max_n = params.get('max_n', 10)  # 扩大测试范围
        self.r_min = params.get('r_min', 1)
        self.r_max = params.get('r_max', 10)
        self.d_min = params.get('d_min', 1)
        self.d_max = params.get('d_max', 10)
        self.a_min = params.get('a_min', 1)
        self.a_max = params.get('a_max', 5)

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        # 保证r1 <= r2 <= r3
        r_samples = sorted([random.randint(self.r_min, self.r_max) for _ in range(3)])
        r1, r2, r3 = r_samples
        d = random.randint(self.d_min, self.d_max)
        a = [random.randint(self.a_min, self.a_max) for _ in range(n)]
        return {
            'n': n,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            'd': d,
            'a': a
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['r1']} {question_case['r2']} {question_case['r3']} {question_case['d']}",
            ' '.join(map(str, question_case['a']))
        ]
        example_input = '\n'.join(input_lines)
        return f"""你是玩家Ziota，需要计算完成游戏《Monster Invaders》的最小时间。游戏规则如下：
- 每个关卡有a_i个普通怪物（1HP）和一个BOSS（2HP）。
- 手枪：伤害1，装填时间r1。
- 激光枪：全体伤害1，装填时间r2。
- AWP：秒杀，装填时间r3。
- 只能同时装填一把枪，初始未装填。
- 必须消灭所有普通怪物后才能用手枪或AWP攻击BOSS。
- 攻击BOSS未击杀会被迫移动到相邻关卡，移动时间d。移动时无法攻击或装填。
- 初始在关卡1，所有BOSS被击杀时游戏结束。

输入格式：
第一行五个整数：n r1 r2 r3 d
第二行n个整数：a_1 a_2 ... a_n

输入实例：
{example_input}

计算最小完成时间，并将答案放入[answer]标签内，例如：[answer]12345[/answer]。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 全部使用整数运算避免浮点误差
            correct = cls.compute_min_time(
                identity['n'], identity['r1'], identity['r2'],
                identity['r3'], identity['d'], identity['a']
            )
            return int(solution) == correct
        except:
            return False

    @staticmethod
    def compute_min_time(n, r1, r2, r3, d, a_list):
        INF = float('inf')
        dp = [[INF]*3 for _ in range(n)]
        ar = a_list  # a values

        # Initialize first level
        dp[0][0] = r1 * ar[0] + r3            # Strategy 1
        dp[0][1] = min(r2 + r1, r1*(ar[0]+2)) # Strategy 2F
        dp[0][2] = INF                        # Strategy 2S (unreachable)

        # Dynamic Programming
        for i in range(n-1):
            next_i = i + 1
            c1 = r1 * ar[next_i] + r3
            c2 = min(r2 + r1, r1*(ar[next_i] + 2))
            
            # 状态转移条件判断
            is_penalty_case = (i == n-2)
            penalty = 2*d if is_penalty_case else 0
            discount = d if is_penalty_case else 0
            
            # 完整状态转移
            # From state 0
            dp[next_i][0] = min(dp[next_i][0], dp[i][0] + c1 + d)
            dp[next_i][1] = min(dp[next_i][1], dp[i][0] + c2 + d + penalty)
            
            # From state 1 (2F)
            new_cost = dp[i][1] + c1 + 3*d - discount
            dp[next_i][0] = min(dp[next_i][0], new_cost)
            dp[next_i][2] = min(dp[next_i][2], dp[i][1] + c2 + 3*d)
            
            # From state 2 (2S)
            dp[next_i][0] = min(dp[next_i][0], dp[i][2] + c1 + d)
            dp[next_i][1] = min(dp[next_i][1], dp[i][2] + c2 + d + penalty)

        return int(min(dp[-1][0], dp[-1][1], dp[-1][2]))  # 取三种状态的最小值
