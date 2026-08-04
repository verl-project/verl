"""# 

### 谜题描述
There is a deck of n cards. The i-th card has a number a_i on the front and a number b_i on the back. Every integer between 1 and 2n appears exactly once on the cards.

A deck is called sorted if the front values are in increasing order and the back values are in decreasing order. That is, if a_i< a_{i+1} and b_i> b_{i+1} for all 1≤ i<n.

To flip a card i means swapping the values of a_i and b_i. You must flip some subset of cards (possibly, none), then put all the cards in any order you like. What is the minimum number of cards you must flip in order to sort the deck?

Input

The first line contains a single integer n (1≤ n≤ 2⋅ 10^5) — the number of cards.

The next n lines describe the cards. The i-th of these lines contains two integers a_i, b_i (1≤ a_i, b_i≤ 2n). Every integer between 1 and 2n appears exactly once.

Output

If it is impossible to sort the deck, output \"-1\". Otherwise, output the minimum number of flips required to sort the deck.

Examples

Input


5
3 10
6 4
1 9
5 8
2 7


Output


2


Input


2
1 2
3 4


Output


-1


Input


3
1 2
3 6
4 5


Output


-1

Note

In the first test case, we flip the cards (1, 9) and (2, 7). The deck is then ordered (3,10), (5,8), (6,4), (7,2), (9,1). It is sorted because 3<5<6<7<9 and 10>8>4>2>1.

In the second test case, it is impossible to sort the deck.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
def main():
    n = int(stdin.readline())
    a = [map(int, stdin.readline().split(), (10, 10)) for _ in xrange(n)]
    m = n * 2
    pa = [0] * m
    f = [0] * m
    d = [0] * m
    for x, y in a:
        x -= 1
        y -= 1
        pa[x] = y
        pa[y] = x
        f[y] = 1
    ans = s = c = 0
    tot = 0
    hi = m - 1
    lo = 0
    ll = rr = -1
    lr = rl = m
    #bl = []
    #br = []
    while tot < n:
        upd = 0
        while hi >= lr:
            if not d[hi]:
                if rl < hi or rr > pa[hi]:
                    print -1
                    return
                upd = 1
                rl = hi
                rr = pa[hi]
                #br.append((rl, rr))
                d[rl] = d[rr] = 1
                s += f[rl]
                c += 1
            hi -= 1
        while lo <= rr:
            if not d[lo]:
                if ll > lo or lr < pa[lo]:
                    print -1
                    return
                upd = 1
                ll = lo
                lr = pa[lo]
                #bl.append((ll, lr))
                d[ll] = d[lr] = 1
                s += f[ll]
                c += 1
            lo += 1
        if not upd:
            ans += min(s, c - s)
            tot += c
            if tot < n:
                ll = lo
                lr = pa[lo]
                #bl.append((ll, lr))
                d[ll] = d[lr] = 1
                lo += 1
                s = f[ll]
                c = 1
    if ll < rl and lr > rr:
        print ans
    else:
        print -1

main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from io import StringIO
import sys

class Dflipthecardsbootcamp(Basebootcamp):
    def __init__(self, max_n=10, min_n=1, n=None):
        super().__init__()
        if n is not None:
            self.n = n
        else:
            self.n = random.randint(min_n, max_n)
    
    def case_generator(self):
        """完全随机生成卡片配对，包含合法和非法案例"""
        n = self.n
        numbers = list(range(1, 2 * n + 1))
        random.shuffle(numbers)
        
        # 确保正确配对生成
        cards = []
        used = set()
        for _ in range(n):
            available = list(set(numbers) - used)
            a = random.choice(available)
            available.remove(a)
            b = random.choice(available)
            cards.append((a, b))
            used.update({a, b})
        
        input_str = f"{n}\n" + "\n".join(f"{a} {b}" for a, b in cards)
        expected_output = self.solve(input_str)
        
        return {
            'n': n,
            'cards': cards,
            'expected_output': expected_output
        }

    @staticmethod
    def solve(input_str):
        """改进的验证算法，修正数组越界问题"""
        original_stdin = sys.stdin
        sys.stdin = StringIO(input_str)
        try:
            n = int(sys.stdin.readline())
            a = []
            for _ in range(n):
                x, y = map(int, sys.stdin.readline().split())
                a.append((x, y))
            
            m = 2 * n  # 正确设置数组大小
            pa = [0] * m
            f = [0] * m
            d = [0] * m
            
            for x, y in a:
                x -= 1
                y -= 1
                if x >= m or y >= m:  # 添加边界检查
                    return -1
                pa[x] = y
                pa[y] = x
                f[y] = 1
            
            ans = s = c = tot = 0
            hi, lo = m - 1, 0
            ll = rr = -1
            lr = rl = m
            
            while tot < n:
                upd = 0
                # 高频错误点修复：添加索引范围检查
                while hi >= max(lr, 0):
                    if hi >= m:  # 防止越界
                        hi = m - 1
                        continue
                    if not d[hi]:
                        if rl < hi or rr > pa[hi]:
                            return -1
                        upd = 1
                        rl, rr = hi, pa[hi]
                        if rl >= m or rr >= m:
                            return -1
                        d[rl] = d[rr] = 1
                        s += f[rl]
                        c += 1
                    hi -= 1
                
                while lo <= min(rr, m-1):
                    if lo < 0:  # 防止负索引
                        lo = 0
                        continue
                    if not d[lo]:
                        if ll > lo or lr < pa[lo]:
                            return -1
                        upd = 1
                        ll, lr = lo, pa[lo]
                        if ll >= m or lr >= m:
                            return -1
                        d[ll] = d[lr] = 1
                        s += f[ll]
                        c += 1
                    lo += 1
                
                if not upd:
                    ans += min(s, c - s)
                    tot += c
                    if tot < n:
                        if lo >= m:  # 处理越界情况
                            return -1
                        try:
                            ll, lr = lo, pa[lo]
                        except IndexError:
                            return -1
                        if ll >= m or lr >= m:
                            return -1
                        d[ll] = d[lr] = 1
                        lo += 1
                        s = f[ll]
                        c = 1
            
            return ans if (ll < rl and lr > rr) else -1
        finally:
            sys.stdin = original_stdin

    @staticmethod
    def prompt_func(question_case):
        # 保持原有prompt格式
        cards_desc = "\n".join([f"Card {i+1}: Front={a}, Back={b}" for i, (a, b) in enumerate(question_case['cards'])])
        return f"""Given a deck of {question_case['n']} cards with unique numbers (1-{2*question_case['n']}) on both sides. Flip some cards and arrange them to satisfy:
- Front numbers strictly increase
- Back numbers strictly decrease

Cards:
{cards_desc}

Output the minimum flips required or -1 if impossible. Put your final answer within [answer] tags like [answer]3[/answer]."""

    @staticmethod
    def extract_output(output):
        # 增强正则匹配，处理多空格情况
        matches = re.findall(r'\[answer\s*\]\s*(-?\d+)\s*\[/answer\s*\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_output']
