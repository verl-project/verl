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
n, a = int(raw_input()), map(int, raw_input().split())
print((\"Bob\" if a.count(min(a)) > n / 2 else \"Alice\"))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ethanosnimbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=50, min_stones=1, max_stones=50):
        # 参数有效性验证与强制偶数处理
        self.min_n = max(2, min_n - (min_n % 2))  # 确保起始为偶数
        self.max_n = min(50, max_n + (max_n % 2))  # 确保结束为偶数
        self.min_n, self.max_n = sorted([self.min_n, self.max_n])
        
        # 处理无效参数场景
        if self.max_n - self.min_n < 2:
            self.min_n, self.max_n = 2, 50

        # 石子数量参数处理
        self.min_stones = max(1, min_stones)
        self.max_stones = max(self.min_stones, min(50, max_stones))

    def case_generator(self):
        # 生成合法测试用例
        possible_n = list(range(self.min_n, self.max_n + 1, 2))
        n = random.choice(possible_n)
        
        # 基础生成逻辑
        a = [random.randint(self.min_stones, self.max_stones) for _ in range(n)]
        
        # 增强边界条件生成（40%概率）
        if random.random() < 0.4:
            target_min = random.randint(self.min_stones, self.max_stones)
            # 确定要设置的最小值数量
            candidate_count = random.choice([
                random.randint(n//2 + 1, n),  # Bob获胜场景
                random.randint(1, n//2)       # Alice获胜场景
            ])
            # 选择要设置的索引
            indices = random.sample(range(n), k=candidate_count)
            # 设置目标最小值并确保其他值>=最小值
            for i in indices:
                a[i] = target_min
            for i in range(n):
                if i not in indices and a[i] < target_min:
                    a[i] = target_min
        
        return {'n': n, 'a': a}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        return f"""## 石子游戏胜负判断 ##

当前游戏参数：
- 石子堆数：{n}堆（偶数）
- 各堆石子数：{a}

游戏规则：
1. Alice和Bob轮流操作，Alice先手
2. 每回合必须选择恰好{n//2}个非空堆
3. 每次操作从每个选中堆移除至少1个石子
4. 无法执行合法操作者判负

请根据当前局面判断获胜者，答案必须严格使用以下格式：
[answer]Alice[/answer] 或 [answer]Bob[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强匹配逻辑（处理换行和空格）
        matches = re.findall(r'\[answer\]\s*(\w+)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1].strip().capitalize() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        n = identity['n']
        current_min = min(a)
        min_count = a.count(current_min)
        correct = "Bob" if min_count > n//2 else "Alice"
        return solution.strip().lower() == correct.lower()
