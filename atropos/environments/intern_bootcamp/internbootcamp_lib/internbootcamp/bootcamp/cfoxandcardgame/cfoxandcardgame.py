"""# 

### 谜题描述
Fox Ciel is playing a card game with her friend Fox Jiro. There are n piles of cards on the table. And there is a positive integer on each card.

The players take turns and Ciel takes the first turn. In Ciel's turn she takes a card from the top of any non-empty pile, and in Jiro's turn he takes a card from the bottom of any non-empty pile. Each player wants to maximize the total sum of the cards he took. The game ends when all piles become empty.

Suppose Ciel and Jiro play optimally, what is the score of the game?

Input

The first line contain an integer n (1 ≤ n ≤ 100). Each of the next n lines contains a description of the pile: the first integer in the line is si (1 ≤ si ≤ 100) — the number of cards in the i-th pile; then follow si positive integers c1, c2, ..., ck, ..., csi (1 ≤ ck ≤ 1000) — the sequence of the numbers on the cards listed from top of the current pile to bottom of the pile.

Output

Print two integers: the sum of Ciel's cards and the sum of Jiro's cards if they play optimally.

Examples

Input

2
1 100
2 1 10


Output

101 10


Input

1
9 2 8 6 5 9 4 7 1 3


Output

30 15


Input

3
3 1 3 2
3 5 4 6
2 8 7


Output

18 18


Input

3
3 1000 1000 1000
6 1000 1000 1000 1000 1000 1000
5 1000 1000 1000 1000 1000


Output

7000 7000

Note

In the first example, Ciel will take the cards with number 100 and 1, Jiro will take the card with number 10.

In the second example, Ciel will take cards with numbers 2, 8, 6, 5, 9 and Jiro will take cards with numbers 4, 7, 1, 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def main():
    n = int(raw_input())
    a = [0, 0]
    c = []
    for i in xrange(n):
        l = map(int, raw_input().split())
        m, l = l[0], l[1:]
        a[0] += sum(l[:m/2])
        a[1] += sum(l) - sum(l[:m-m/2])
        if m % 2:
            c.append(l[m/2])
    c.sort(reverse=True)
    for i, x in enumerate(c):
        a[i%2] += x
    print a[0], a[1]
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cfoxandcardgamebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'min_n': params.get('min_n', 1),
            'max_n': params.get('max_n', 10),
            'min_pile_size': params.get('min_pile_size', 1),
            'max_pile_size': params.get('max_pile_size', 10),
            'min_card_value': params.get('min_card_value', 1),
            'max_card_value': params.get('max_card_value', 100),
        }

    def case_generator(self):
        n = random.randint(self.params['min_n'], self.params['max_n'])
        piles = []
        for _ in range(n):
            s_i = random.randint(self.params['min_pile_size'], self.params['max_pile_size'])
            cards = [random.randint(self.params['min_card_value'], self.params['max_card_value']) for _ in range(s_i)]
            piles.append(cards)
        return {'piles': piles}

    @staticmethod
    def prompt_func(question_case) -> str:
        piles = question_case['piles']
        prompt = "Fox Ciel和Fox Jiro正在玩一个卡片游戏。桌上有n个堆的卡片，每个卡片上有一个正整数。玩家轮流取牌，Ciel先手。在她的回合，她可以从任意非空堆顶部取一张牌；Jiro在自己的回合从任意非空堆底部取牌。双方都采取最优策略来最大化自己的总得分。游戏结束时，输出两人的得分。\n\n输入格式：\n第一行为整数n（1 ≤ n ≤ 100），接下来n行每行描述一个堆：第一个数s_i（1 ≤ s_i ≤ 100）表示卡片数，随后是s_i个正整数（从顶到底）。\n\n当前问题输入数据：\n"
        prompt += f"{len(piles)}\n"
        for pile in piles:
            prompt += f"{len(pile)} {' '.join(map(str, pile))}\n"
        prompt += "\n请输出两个整数（Ciel和Jiro的得分），用空格分隔，放在[answer]和[/answer]之间。\n示例：[answer]100 50[/answer]"
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        parts = last_match.split()
        if len(parts) != 2:
            return None
        try:
            return (int(parts[0]), int(parts[1]))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = [0, 0]
        c = []
        piles = identity['piles']
        for pile in piles:
            m = len(pile)
            a[0] += sum(pile[:m//2])
            a[1] += sum(pile) - sum(pile[:m - m//2])
            if m % 2:
                c.append(pile[m//2])
        c.sort(reverse=True)
        for i, x in enumerate(c):
            a[i % 2] += x
        
        if not (isinstance(solution, (tuple, list)) and len(solution) == 2):
            return False
        try:
            return solution[0] == a[0] and solution[1] == a[1]
        except:
            return False
