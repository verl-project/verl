"""# 

### 谜题描述
Omkar is playing his favorite pixelated video game, Bed Wars! In Bed Wars, there are n players arranged in a circle, so that for all j such that 2 ≤ j ≤ n, player j - 1 is to the left of the player j, and player j is to the right of player j - 1. Additionally, player n is to the left of player 1, and player 1 is to the right of player n.

Currently, each player is attacking either the player to their left or the player to their right. This means that each player is currently being attacked by either 0, 1, or 2 other players. A key element of Bed Wars strategy is that if a player is being attacked by exactly 1 other player, then they should logically attack that player in response. If instead a player is being attacked by 0 or 2 other players, then Bed Wars strategy says that the player can logically attack either of the adjacent players.

Unfortunately, it might be that some players in this game are not following Bed Wars strategy correctly. Omkar is aware of whom each player is currently attacking, and he can talk to any amount of the n players in the game to make them instead attack another player — i. e. if they are currently attacking the player to their left, Omkar can convince them to instead attack the player to their right; if they are currently attacking the player to their right, Omkar can convince them to instead attack the player to their left. 

Omkar would like all players to be acting logically. Calculate the minimum amount of players that Omkar needs to talk to so that after all players he talked to (if any) have changed which player they are attacking, all players are acting logically according to Bed Wars strategy.

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 10^4). The descriptions of the test cases follows.

The first line of each test case contains one integer n (3 ≤ n ≤ 2 ⋅ 10^5) — the amount of players (and therefore beds) in this game of Bed Wars.

The second line of each test case contains a string s of length n. The j-th character of s is equal to L if the j-th player is attacking the player to their left, and R if the j-th player is attacking the player to their right.

It is guaranteed that the sum of n over all test cases does not exceed 2 ⋅ 10^5.

Output

For each test case, output one integer: the minimum number of players Omkar needs to talk to to make it so that all players are acting logically according to Bed Wars strategy.

It can be proven that it is always possible for Omkar to achieve this under the given constraints.

Example

Input


5
4
RLRL
6
LRRRRL
8
RLLRRRLL
12
LLLLRRLRRRLL
5
RRRRR


Output


0
1
1
3
2

Note

In the first test case, players 1 and 2 are attacking each other, and players 3 and 4 are attacking each other. Each player is being attacked by exactly 1 other player, and each player is attacking the player that is attacking them, so all players are already being logical according to Bed Wars strategy and Omkar does not need to talk to any of them, making the answer 0.

In the second test case, not every player acts logically: for example, player 3 is attacked only by player 2, but doesn't attack him in response. Omkar can talk to player 3 to convert the attack arrangement to LRLRRL, in which you can see that all players are being logical according to Bed Wars strategy, making the answer 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from itertools import groupby
def solve():
    n = int(stdin.readline())
    s = stdin.readline().strip()
    if s == s[0] * n:
        ans = (n + 2) / 3
    else:
        t = [(k, len(list(v))) for k, v in groupby(s)]
        if t[0][0] == t[-1][0]:
            t[0] = (t[0][0], t[0][1] + t[-1][1])
            t.pop()
        ans = 0
        for k, v in t:
            ans += v / 3
    print ans

T = int(stdin.readline())
for _ in xrange(T):
    solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
from itertools import groupby
import re

class Domkarandbedwarsbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=20):
        """
        初始化训练场参数，确保n ≥ 3
        参数:
            min_n (int): 最小玩家数 ≥3
            max_n (int): 最大玩家数
        """
        assert min_n >= 3, "玩家数至少为3"
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        """生成合法测试用例及预计算答案"""
        n = random.randint(self.min_n, self.max_n)
        # 生成有效攻击序列
        s = []
        for _ in range(n):
            s.append(random.choice(['L', 'R']))
        s = ''.join(s)
        
        # 正确解法逻辑
        if all(c == s[0] for c in s):
            ans = (n + 2) // 3
        else:
            groups = []
            for k, grp in groupby(s):
                groups.append( (k, sum(1 for _ in grp)) )
            
            # 合并循环相同段
            if len(groups) > 1 and groups[0][0] == groups[-1][0]:
                groups[0] = (groups[0][0], groups[0][1] + groups[-1][1])
                groups.pop()
            
            ans = sum( cnt // 3 for _, cnt in groups )
        
        return {
            'n': n,
            's': s,
            'correct_answer': ans
        }
    
    @staticmethod
    def prompt_func(case) -> str:
        """生成详细规则描述的问题模板"""
        return f"""## Bed Wars策略分析

**游戏配置**
- 玩家总数：{case['n']}
- 攻击方向序列：`{case['s']}`（索引从1开始，L/R表示攻击方向）

**策略规则**
1. 玩家被1人攻击时必须反击攻击者
2. 被0或2人攻击时可自由选择攻击方向
3. 每次转换可改变一个玩家的攻击方向

**任务**
计算使得所有玩家符合策略的最小转换次数，将最终数字包裹在[answer]标签内，如：[answer]3[/answer]"""
    
    @staticmethod
    def extract_output(text: str):
        """严格提取最后一个答案标签内容"""
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', text, re.IGNORECASE)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """直接比对预计算答案"""
        return solution == identity['correct_answer']
