"""# 

### 谜题描述
In some country live wizards. They love playing with numbers. 

The blackboard has two numbers written on it — a and b. The order of the numbers is not important. Let's consider a ≤ b for the sake of definiteness. The players can cast one of the two spells in turns:

  * Replace b with b - ak. Number k can be chosen by the player, considering the limitations that k > 0 and b - ak ≥ 0. Number k is chosen independently each time an active player casts a spell. 
  * Replace b with b mod a. 



If a > b, similar moves are possible.

If at least one of the numbers equals zero, a player can't make a move, because taking a remainder modulo zero is considered somewhat uncivilized, and it is far too boring to subtract a zero. The player who cannot make a move, loses.

To perform well in the magic totalizator, you need to learn to quickly determine which player wins, if both wizards play optimally: the one that moves first or the one that moves second.

Input

The first line contains a single integer t — the number of input data sets (1 ≤ t ≤ 104). Each of the next t lines contains two integers a, b (0 ≤ a, b ≤ 1018). The numbers are separated by a space.

Please do not use the %lld specificator to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specificator.

Output

For any of the t input sets print \"First\" (without the quotes) if the player who moves first wins. Print \"Second\" (without the quotes) if the player who moves second wins. Print the answers to different data sets on different lines in the order in which they are given in the input. 

Examples

Input

4
10 21
31 10
0 1
10 30


Output

First
Second
Second
First

Note

In the first sample, the first player should go to (11,10). Then, after a single move of the second player to (1,10), he will take 10 modulo 1 and win.

In the second sample the first player has two moves to (1,10) and (21,10). After both moves the second player can win.

In the third sample, the first player has no moves.

In the fourth sample, the first player wins in one move, taking 30 modulo 10.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
from math import *

def win(a,b):
    if (a==0):
        return False
    if (b==0):
        return False
    if (not win(b%a,a)):
        return True
    ans=b//a
    ans%=a+1
    ans%=2
    if (ans%2==1):
        return False
    else:
        return True

try:
    fi = open(\"input.txt\", \"r\")
    fo = open(\"output.txt\", \"w\")
except:
    fi = sys.stdin
    fo = sys.stdout

tests=int(fi.readline())
for test in range(tests):
    a,b=map(int,fi.readline().split())
    if (win(min(a,b),max(a,b))):
        fo.write(\"First\n\")
    else:
        fo.write(\"Second\n\")
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
from functools import lru_cache

class Cwizardsandnumbersbootcamp(Basebootcamp):
    def __init__(self, a_min=0, a_max=10**5, b_min=0, b_max=10**5):
        super().__init__()
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max

    def case_generator(self):
        a = random.randint(self.a_min, self.a_max)
        b = random.randint(self.b_min, self.b_max)
        a, b = sorted((a, b))
        correct_answer = 'First' if self.is_first_win(a, b) else 'Second'
        return {
            'a': a,
            'b': b,
            'correct_answer': correct_answer
        }

    @staticmethod
    @lru_cache(maxsize=None)
    def is_first_win(a, b):
        if a == 0:
            return False
        if not Cwizardsandnumbersbootcamp.is_first_win(b % a, a):
            return True
        ans = b // a
        ans %= a + 1
        ans %= 2
        return ans % 2 == 0

    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        b = question_case['b']
        prompt = f"两个巫师在玩一个数字游戏。黑板上写着两个数a={a}和b={b}。玩家轮流进行以下操作：\n"
        prompt += "1. 将较大的数减去较小的数的k倍（k>0，结果不能为负）。\n"
        prompt += "2. 将较大的数对较小的数取模。\n"
        prompt += "无法进行操作的玩家输。轮到你时，作为先手，你会赢吗？\n"
        prompt += "请判断先手会赢还是输，输出'First'表示先手赢，'Second'表示先手输。请将答案放在[answer]标签中，例如：[answer]First[/answer]\n"
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if matches:
            return matches[-1].strip()
        else:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_answer = identity['correct_answer']
        return solution == correct_answer
