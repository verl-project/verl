"""# 

### 谜题描述
There are n players sitting at a round table. All of them have s cards of n colors in total. Besides, initially the first person had cards of only the first color, the second one had cards of only the second color and so on. They can swap the cards by the following rules: 

  * as the players swap, a player can give a card of his color only; 
  * a player can't accept a card of a color he already has (particularly, he can't take cards of his color, no matter whether he has given out all of them or not); 
  * during one swap a pair of people swaps cards (each person gives one card and takes one card). 



The aim of all n people is as follows: each of them should give out all the cards he had initially (that is, all cards of his color). Your task is to denote whether such sequence of swaps is possible. If the answer is positive, you should list all the swaps.

Input

The first line contains integers n (1 ≤ n ≤ 200000) and s (1 ≤ s ≤ 200000). The second line contains n numbers, the i-th number stands for how many cards the i-th player has by the moment the game starts. It is possible that a player has no cards initially.

Output

On the first line print \"No\" if such sequence of swaps is impossible. Otherwise, print \"Yes\". If the answer is positive, next print number k — the number of the swaps. Then on k lines describe the swaps by pairs of indices of the swapping players. Print the swaps and the numbers of the swaps in any order.

Examples

Input

4 8
2 2 2 2


Output

Yes
4
4 3
4 2
1 3
1 2


Input

6 12
1 1 2 2 3 3


Output

Yes
6
6 5
6 4
6 3
5 4
5 3
2 1


Input

5 5
0 0 0 0 5


Output

No

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from heapq import *
n, s = map(int, raw_input().split())
a = map(int, raw_input().split())
q = [(-x, i) for (i, x) in enumerate(a) if x > 0]
heapify(q)
res = []
while q:
    (x, i) = heappop(q)
    if -x > len(q):
        print 'No'
        break
    for (y, j) in [heappop(q) for _ in xrange(-x)]:
        res.append((i + 1, j + 1))
        if y + 1: heappush(q, (y + 1, j))
else:
    print 'Yes'
    print len(res)
    for (i, j) in res: print i, j
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from heapq import heapify, heappop, heappush

class Cswapsbootcamp(Basebootcamp):
    def __init__(self, n_range=(1, 10), s_range=(1, 20)):
        self.n_range = n_range
        self.s_range = s_range
        
    def case_generator(self):
        n = random.randint(*self.n_range)
        s = random.randint(*self.s_range)
        a = self.generate_a(n, s)
        possible = self.check_possible(n, s, a)
        return {'n': n, 's': s, 'a': a, 'possible': possible}
    
    @staticmethod
    def generate_a(n, s):
        if n == 0:
            return []
        a = [0] * n
        total = 0
        for i in range(n-1):
            max_assign = s - total
            a[i] = random.randint(0, max_assign)
            total += a[i]
        a[-1] = s - total
        return a
    
    @staticmethod
    def check_possible(n, s, a):
        q = [(-x, i) for i, x in enumerate(a) if x > 0]
        heapify(q)
        try:
            while q:
                x, i = heappop(q)
                if -x > len(q):
                    return False
                partners = []
                for _ in range(-x):
                    if not q:
                        return False
                    partners.append(heappop(q))
                for y, j in partners:
                    if y + 1 != 0:
                        heappush(q, (y + 1, j))
            return True
        except:
            return False
    
    @staticmethod
    def prompt_func(case):
        n = case['n']
        s = case['s']
        a = case['a']
        problem_text = (
            f"有 {n} 个玩家围坐在圆桌旁，初始时每个玩家拥有特定颜色的卡牌。玩家可以按照以下规则交换卡牌：\n"
            "1. 每次交换必须是一对玩家互相交换\n"
            "2. 每个玩家只能给出自己初始颜色的卡牌\n"
            "3. 玩家不能接受自己已拥有颜色的卡牌（包括初始颜色）\n"
            "目标：所有玩家通过若干次交换后，必须给出自己所有的初始颜色卡牌。\n\n"
            "输入格式：\n"
            "第一行两个整数 n 和 s（玩家数量和初始卡牌总数）\n"
            "第二行 n 个整数表示每个玩家的初始卡牌数\n\n"
            "当前问题实例：\n"
            f"{n} {s}\n"
            f"{' '.join(map(str, a))}\n\n"
            "请判断是否存在合法交换方案。若存在，输出'Yes'并给出交换步骤；否则输出'No'。\n"
            "答案请用[answer]标签包裹，示例：\n"
            "[answer]\nYes\n3\n1 2\n2 3\n3 1\n[/answer]"
        )
        return problem_text
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        lines = solution.strip().split('\n')
        expected_possible = identity['possible']
        if not expected_possible:
            return len(lines) >= 1 and lines[0].strip() == "No"
        if len(lines) < 2 or lines[0].strip() != "Yes":
            return False
        try:
            k = int(lines[1].strip())
            if k != identity['s'] // 2 or len(lines) != 2 + k:
                return False
            swaps = [tuple(map(int, line.strip().split())) for line in lines[2:]]
            swap_counts = [0] * identity['n']
            for i, j in swaps:
                if i == j or not (1 <= i <= identity['n']) or not (1 <= j <= identity['n']):
                    return False
                swap_counts[i-1] += 1
                swap_counts[j-1] += 1
            if swap_counts != identity['a']:
                return False
            players = [{i+1: identity['a'][i]} if identity['a'][i] > 0 else {} for i in range(identity['n'])]
            for i, j in swaps:
                if (i not in players[i-1]) or players[i-1][i] < 1 or (j not in players[j-1]) or players[j-1][j] < 1:
                    return False
                players[i-1][i] -= 1
                if players[i-1][i] == 0:
                    del players[i-1][i]
                players[j-1][j] -= 1
                if players[j-1][j] == 0:
                    del players[j-1][j]
                players[i-1][j] = players[i-1].get(j, 0) + 1
                players[j-1][i] = players[j-1].get(i, 0) + 1
            for idx in range(identity['n']):
                if (idx + 1) in players[idx]:
                    return False
            return True
        except:
            return False
