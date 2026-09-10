"""# 

### 谜题描述
Two bored soldiers are playing card war. Their card deck consists of exactly n cards, numbered from 1 to n, all values are different. They divide cards between them in some manner, it's possible that they have different number of cards. Then they play a \"war\"-like card game. 

The rules are following. On each turn a fight happens. Each of them picks card from the top of his stack and puts on the table. The one whose card value is bigger wins this fight and takes both cards from the table to the bottom of his stack. More precisely, he first takes his opponent's card and puts to the bottom of his stack, and then he puts his card to the bottom of his stack. If after some turn one of the player's stack becomes empty, he loses and the other one wins. 

You have to calculate how many fights will happen and who will win the game, or state that game won't end.

Input

First line contains a single integer n (2 ≤ n ≤ 10), the number of cards.

Second line contains integer k1 (1 ≤ k1 ≤ n - 1), the number of the first soldier's cards. Then follow k1 integers that are the values on the first soldier's cards, from top to bottom of his stack.

Third line contains integer k2 (k1 + k2 = n), the number of the second soldier's cards. Then follow k2 integers that are the values on the second soldier's cards, from top to bottom of his stack.

All card values are different.

Output

If somebody wins in this game, print 2 integers where the first one stands for the number of fights before end of game and the second one is 1 or 2 showing which player has won.

If the game won't end and will continue forever output  - 1.

Examples

Input

4
2 1 3
2 4 2


Output

6 2

Input

3
1 2
2 1 3


Output

-1

Note

First sample: 

<image>

Second sample: 

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def myHash(n, a, b):
    h = n
    for i in a:
        h = 10 * h + i
    for i in b:
        h = 10 * h + i
        
    return h

def passTurn(n, m, a, b):
    if(a[0] > b[0]):
        n += 1
        m -= 1
        a.append(b[0])
        a.append(a[0])
        del a[0]
        del b[0]
    else:
        n -= 1
        m += 1
        b.append(a[0])
        b.append(b[0])
        del a[0]
        del b[0]

    return n, m, a, b

tot = int(raw_input())
lis = [int(x) for x in raw_input().split()]
n, a = lis[0], lis[1:]
lis = [int(x) for x in raw_input().split()]
m, b = lis[0], lis[1:]


s = set([myHash(n, a, b)])

turn = 0
win  = 0

while True:
    n, m, a, b = passTurn(n, m, a, b)
    h = myHash(n, a, b)
    if h in s:
        break
    else:
        s.add(h)
    turn += 1
    if m == 0:
        win = 1
        break
    elif n == 0:
        win = 2
        break

if win == 0:
    print(-1)
else:
    print \"{0} {1}\".format(turn, win)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Csoldierandcardsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_n = params.get('min_n', 2)
        self.max_n = params.get('max_n', 10)
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        cards = list(range(1, n+1))
        random.shuffle(cards)
        k1 = random.randint(1, n-1)
        return {
            'n': n,
            'player1': cards[:k1],
            'player2': cards[k1:]
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        prompt = f"""Two soldiers are playing a card war game. The deck has {question_case['n']} unique cards numbered 1 to {question_case['n']}. 

Soldier 1 has {len(question_case['player1'])} cards (top to bottom): {', '.join(map(str, question_case['player1']))}.
Soldier 2 has {len(question_case['player2'])} cards (top to bottom): {', '.join(map(str, question_case['player2']))}.

Rules:
1. Each fight: Both play their top card. Higher value wins.
2. Winner takes both cards (opponent's first, then theirs) to their deck's bottom.
3. Game ends when a soldier has no cards. If a state repeats, the game loops infinitely.

You must output either:
- The number of fights and the winner (e.g., [answer]6 2[/answer])
- Or [answer]-1[/answer] if it never ends.

Use exact format with [answer] tags."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_str = matches[-1].strip()
        if answer_str == '-1':
            return -1
        parts = answer_str.split()
        if len(parts) != 2:
            return None
        try:
            fights = int(parts[0])
            winner = int(parts[1])
            if winner in (1, 2) and fights >= 0:
                return (fights, winner)
        except:
            pass
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['player1'].copy()
        b = identity['player2'].copy()
        k1, k2 = len(a), len(b)
        states = set()
        
        def my_hash(k1_val, k2_val, a_list, b_list):
            h = k1_val * 100 + k2_val
            for c in a_list:
                h = h * 10 + c
            for c in b_list:
                h = h * 10 + c
            return h
        
        current_hash = my_hash(k1, k2, a, b)
        states.add(current_hash)
        win = 0
        turn = 0
        
        while True:
            if k1 == 0:
                win = 2
                break
            if k2 == 0:
                win = 1
                break
            
            card1, card2 = a[0], b[0]
            if card1 > card2:
                a.append(b.pop(0))
                a.append(a.pop(0))
                k1 += 1
                k2 -= 1
            else:
                b.append(a.pop(0))
                b.append(b.pop(0))
                k2 += 1
                k1 -= 1
            
            new_hash = my_hash(k1, k2, a, b)
            if new_hash in states:
                win = 0
                break
            states.add(new_hash)
            turn += 1
            
            if k1 == 0 or k2 == 0:
                win = 1 if k2 == 0 else 2
                break
        
        correct = (turn, win) if win else -1
        return solution == correct
