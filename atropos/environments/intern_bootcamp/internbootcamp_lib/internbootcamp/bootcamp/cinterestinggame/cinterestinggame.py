"""# 

### 谜题描述
Two best friends Serozha and Gena play a game.

Initially there is one pile consisting of n stones on the table. During one move one pile should be taken and divided into an arbitrary number of piles consisting of a1 > a2 > ... > ak > 0 stones. The piles should meet the condition a1 - a2 = a2 - a3 = ... = ak - 1 - ak = 1. Naturally, the number of piles k should be no less than two.

The friends play in turns. The player who cannot make a move loses. Serozha makes the first move. Who will win if both players play in the optimal way?

Input

The single line contains a single integer n (1 ≤ n ≤ 105).

Output

If Serozha wins, print k, which represents the minimal number of piles into which he can split the initial one during the first move in order to win the game.

If Gena wins, print \"-1\" (without the quotes).

Examples

Input

3


Output

2


Input

6


Output

-1


Input

100


Output

8

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()


spg = [0] * (n + 1)
xor = [0] * (n + 1)

for i in range(3, n + 1):
    k = 2
    movs = set()
    while k * (k + 1) <= 2 * i:
        s = 2 * i - k * (k - 1)

        if s % (2 * k) == 0:
            a = s / 2 / k
            movs.add(xor[a + k - 1] ^ xor[a - 1])
        k += 1

    mex = 0
    while mex in movs:
        mex += 1
    spg[i] = mex
    xor[i] = xor[i - 1] ^ mex

if spg[n]:
    k = 2
    while k * (k + 1) <= 2 * i:
        s = 2 * i - k * (k - 1)

        if s % (2 * k) == 0:
            a = s / 2 / k
            if (xor[a + k - 1] ^ xor[a - 1]) == 0:
                break
        k += 1
    print(k)
else:
    print(-1)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cinterestinggamebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100000):
        self.min_n = min_n
        self.max_n = max_n

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        correct_answer = self._solve(n)
        return {"n": n, "correct_answer": correct_answer}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        prompt = f"""你是石子游戏专家，请解决以下问题：

游戏规则：
1. 初始有一堆石子，共{n}个。
2. 玩家轮流操作，每次操作必须选择一堆石子，并将其分成k堆（k≥2），且满足各堆石子数目严格递减且相邻两堆数目差为1。
3. 无法操作的玩家输。Serozha先手，两人都采取最优策略。

你的任务：确定Serozha是否能赢。若赢，输出他第一次分割的最小k值；否则输出-1。

输入格式：整数n={n}

输出格式：答案放在[answer]和[/answer]之间，例如[answer]2[/answer]。

请仔细思考，给出正确的答案。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            user_answer = int(solution)
            return user_answer == identity['correct_answer']
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _solve(n):
        if n == 1:
            return -1
        
        spg = [0] * (n + 1)
        xor = [0] * (n + 1)
        
        for i in range(3, n + 1):
            movs = set()
            k = 2
            while k * (k + 1) <= 2 * i:
                s = 2 * i + k * (k - 1)
                if s % (2 * k) == 0:
                    a = s // (2 * k)
                    if a >= k:  # 确保分割后的堆数满足严格递减条件
                        xor_total = xor[a] ^ xor[a - k]
                        movs.add(xor_total)
                k += 1
            
            mex = 0
            while mex in movs:
                mex += 1
            spg[i] = mex
            xor[i] = xor[i - 1] ^ spg[i]
        
        if spg[n] == 0:
            return -1
        else:
            min_k = None
            k = 2
            while k * (k + 1) <= 2 * n:
                s = 2 * n + k * (k - 1)
                if s % (2 * k) == 0:
                    a = s // (2 * k)
                    if a >= k:
                        xor_total = xor[a] ^ xor[a - k]
                        if xor_total == 0:
                            min_k = k
                            break
                k += 1
            return min_k if min_k is not None else -1
