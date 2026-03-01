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
from operator import __xor__

known = {1:0, 2:0}
win = {1:-1, 2:-1}

def split(stone, k):
    n = int( ( stone - (k*(k-1)/2)) / k)
    if n <= 0 or k*n + k*(k-1)/2 != stone:
        return []
    return range(n, n+k)

def mex(iterable):
    i = 0
    while i in iterable:
        i += 1
    return i

def g(stone):
    if stone in known:
        return known[stone]
    loc = set()
    win_k = -1
    for k in range(2, int((2*stone) ** .5) + 1):
        sp =  split(stone, k)
        if sp:
            xor = reduce(__xor__, map(g, sp), 0)
            loc.add(xor)
            if xor == 0 and win_k == -1:
                win_k = k
    xor = mex(loc)
    known[stone] = xor
    win[stone] = win_k
    return xor

if __name__ == \"__main__\":
    n = int(raw_input().strip())
    for i in range(10**5): g(n)
    print win[n]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from functools import reduce
from operator import xor
import re
import random
from bootcamp import Basebootcamp

class Einterestinggamebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100000):
        self.min_n = min_n
        self.max_n = max_n
        self.win_dict = {}
        self.precompute()
    
    def split(self, stone, k):
        total = k * (k - 1) // 2
        numerator = stone + total
        if numerator % k != 0:
            return []
        a = numerator // k
        if a < k:
            return []
        piles = [a - i for i in range(k)]
        if sum(piles) != stone:
            return []
        return piles

    def mex(self, s):
        i = 0
        while i in s:
            i += 1
        return i

    def precompute(self):
        self.win_dict = {1: -1, 2: -1}
        known = {1: 0, 2: 0}
        for stone in range(3, self.max_n + 1):
            mex_set = set()
            win_k = -1
            max_k = int((2 * stone) ** 0.5) + 1
            for k in range(2, max_k + 1):
                piles = self.split(stone, k)
                if not piles:
                    continue
                try:
                    xor_val = reduce(xor, (known[p] for p in piles))
                except KeyError:
                    continue
                mex_set.add(xor_val)
                if xor_val == 0 and win_k == -1:
                    win_k = k
            mex_val = self.mex(mex_set)
            known[stone] = mex_val
            self.win_dict[stone] = win_k

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        return {'n': n, 'correct_answer': self.win_dict.get(n, -1)}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""Two friends Serozha and Gena play a stone splitting game. Initially there is 1 pile of {n} stones. On each turn, a player splits a pile into ≥2 strictly decreasing piles with consecutive differences of 1. The player who cannot move loses. Serozha goes first.

If Serozha can win with optimal play, find the minimal k (number of piles he splits into on his first move). Otherwise, output -1. 

Format your answer as [answer]k[/answer] where k is -1 or an integer ≥2.

Examples:
Input: 3 → Output: [answer]2[/answer]
Input: 6 → Output: [answer]-1[/answer]
Input: 100 → Output: [answer]8[/answer]
Your task: {n}"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer', -1)
