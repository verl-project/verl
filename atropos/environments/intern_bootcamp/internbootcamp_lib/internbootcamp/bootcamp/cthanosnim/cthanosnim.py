"""# 

### 谜题描述
Alice and Bob are playing a game with n piles of stones. It is guaranteed that n is an even number. The i-th pile has a_i stones.

Alice and Bob will play a game alternating turns with Alice going first.

On a player's turn, they must choose exactly n/2 nonempty piles and independently remove a positive number of stones from each of the chosen piles. They can remove a different number of stones from the piles in a single turn. The first player unable to make a move loses (when there are less than n/2 nonempty piles).

Given the starting configuration, determine who will win the game.

Input

The first line contains one integer n (2 ≤ n ≤ 50) — the number of piles. It is guaranteed that n is an even number.

The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 50) — the number of stones in the piles.

Output

Print a single string \"Alice\" if Alice wins; otherwise, print \"Bob\" (without double quotes).

Examples

Input


2
8 8


Output


Bob


Input


4
3 1 4 1


Output


Alice

Note

In the first example, each player can only remove stones from one pile (2/2=1). Alice loses, since Bob can copy whatever Alice does on the other pile, so Alice will run out of moves first.

In the second example, Alice can remove 2 stones from the first pile and 3 stones from the third pile on her first move to guarantee a win.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())

piles = [int(x) for x in raw_input().split()]

piles.sort()

if piles[0] == piles[n/2]:
    print(\"Bob\")
else:
    print(\"Alice\")
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cthanosnimbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=50, **params):
        super().__init__()
        self.min_n = max(2, min_n)
        self.max_n = min(50, max_n)
        possible_n = [n for n in range(self.min_n, self.max_n + 1) if n % 2 == 0]
        if not possible_n:
            raise ValueError("No valid even n in the given range")
        self.possible_n = possible_n

    def case_generator(self):
        n = random.choice(self.possible_n)
        mid_index = n // 2
        
        # Ensure max(left) +1 <=50 for Alice case
        if random.random() < 0.5:
            # Generate Bob case (sorted[0] == sorted[mid_index])
            base = random.randint(1, 50)
            piles = [base] * (mid_index + 1) + [random.randint(base, 50) for _ in range(n - mid_index - 1)]
        else:
            # Generate Alice case with safe value range
            max_left = random.randint(1, 49)
            left = [random.randint(1, max_left) for _ in range(mid_index)]
            right = [random.randint(max_left + 1, 50) for _ in range(n - mid_index)]
            piles = left + right
        
        random.shuffle(piles)
        return {
            'n': n,
            'piles': piles
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        piles = question_case['piles']
        return f"""Alice和Bob正在玩一个石子堆游戏，规则如下：

- 游戏使用{n}个石子堆（保证是偶数）
- 两人轮流操作，Alice先手
- 每次必须选择恰好{n//2}个非空堆，并从每个选中堆移除至少1个石子
- 无法进行合法操作（当剩余非空堆少于{n//2}时）的玩家判负

当前游戏参数：
n = {n}
各堆石子数 = {', '.join(map(str, piles))}

请分析游戏结果并判断胜者。将最终答案放在[answer]标签内，例如：[answer]Alice[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(Alice|Bob)\s*\[/answer\]', output, re.I)
        return matches[-1].capitalize() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        sorted_piles = sorted(identity['piles'])
        n = identity['n']
        mid_index = n // 2
        return solution == ("Bob" if sorted_piles[0] == sorted_piles[mid_index] else "Alice")
