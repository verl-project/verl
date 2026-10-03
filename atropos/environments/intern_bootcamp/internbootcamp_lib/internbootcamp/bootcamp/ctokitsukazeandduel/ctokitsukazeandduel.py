"""# 

### 谜题描述
\"Duel!\"

Betting on the lovely princess Claris, the duel between Tokitsukaze and Quailty has started.

There are n cards in a row. Each card has two sides, one of which has color. At first, some of these cards are with color sides facing up and others are with color sides facing down. Then they take turns flipping cards, in which Tokitsukaze moves first. In each move, one should choose exactly k consecutive cards and flip them to the same side, which means to make their color sides all face up or all face down. If all the color sides of these n cards face the same direction after one's move, the one who takes this move will win.

Princess Claris wants to know who will win the game if Tokitsukaze and Quailty are so clever that they won't make mistakes.

Input

The first line contains two integers n and k (1 ≤ k ≤ n ≤ 10^5).

The second line contains a single string of length n that only consists of 0 and 1, representing the situation of these n cards, where the color side of the i-th card faces up if the i-th character is 1, or otherwise, it faces down and the i-th character is 0.

Output

Print \"once again\" (without quotes) if the total number of their moves can exceed 10^9, which is considered a draw.

In other cases, print \"tokitsukaze\" (without quotes) if Tokitsukaze will win, or \"quailty\" (without quotes) if Quailty will win.

Note that the output characters are case-sensitive, and any wrong spelling would be rejected.

Examples

Input


4 2
0101


Output


quailty


Input


6 1
010101


Output


once again


Input


6 5
010101


Output


tokitsukaze


Input


4 1
0011


Output


once again

Note

In the first example, no matter how Tokitsukaze moves, there would be three cards with color sides facing the same direction after her move, and Quailty can flip the last card to this direction and win.

In the second example, no matter how Tokitsukaze moves, Quailty can choose the same card and flip back to the initial situation, which can allow the game to end in a draw.

In the third example, Tokitsukaze can win by flipping the leftmost five cards up or flipping the rightmost five cards down.

The fourth example can be explained in the same way as the second example does.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

first = [1e9]*2
last = [0]*2

n,k = [int(x) for x in input().split()]
for i,a in enumerate(int(x) for x in input()):
    first[a] = min(first[a], i)
    last[a] = i

for a in [0,1]:
    if last[a] - first[a] + 1 <= k:
        print 'tokitsukaze'
        sys.exit()

for a in [0,1]:
    if last[a] - first[a] + 1 > k + 1 or \
        first[a] >= k or n - 1 - last[a] >= k:
        print 'once again'
        sys.exit()
print 'quailty'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def validate_case(n, k, s):
    first = [n] * 2
    last = [-1] * 2
    
    for i in range(n):
        a = int(s[i])
        first[a] = min(first[a], i)
        last[a] = max(last[a], i)
    
    # Check immediate win for 0 or 1
    for a in [0, 1]:
        if first[a] <= last[a] and (last[a] - first[a] + 1) <= k:
            return 'tokitsukaze'
    
    # Check draw conditions
    for a in [0, 1]:
        if first[a] > last[a]:
            continue
        
        left_space = first[a]
        right_space = (n-1) - last[a]
        len_a = last[a] - first[a] + 1
        
        if len_a > (k+1) or left_space >= k or right_space >= k:
            return 'once again'
    
    return 'quailty'

class Ctokitsukazeandduelbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=20, **kwargs):
        self.min_n = max(2, min_n)  # Prevent n=1 edge case
        self.max_n = max_n
        self.kwargs = kwargs
    
    def case_generator(self):
        while True:
            n = random.randint(self.min_n, self.max_n)
            k = random.randint(1, n)
            s = ''.join(random.choice('01') for _ in range(n))
            
            # Allow all-0 or all-1 cases
            if validate_case(n, k, s) in ['tokitsukaze', 'quailty', 'once again']:
                return {'n': n, 'k': k, 's': s}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        s = question_case['s']
        prompt = f"""## Ctokitsukazeandduel Game Rules
- {n} cards arranged in a row with states: {s} (1=UP, 0=DOWN)
- Players alternate turns (Tokitsukaze first)
- Each turn: flip exactly {k} consecutive cards to same state
- Immediate win if all cards match after move
- 1,000,000,000+ moves = draw

## Your Task
Analyze the initial configuration and determine the game outcome. Put your final answer (exactly one of these) between [answer] tags:
[answer]tokitsukaze[/answer]  
[answer]quailty[/answer]  
[answer]once again[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL|re.IGNORECASE)
        return matches[-1].strip().lower() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            k = identity['k']
            s = identity['s']
            correct = validate_case(n, k, s)
            return solution == correct.lower()
        except:
            return False
