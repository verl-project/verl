"""# 

### 谜题描述
After all the events in Orlando we all know, Sasha and Roma decided to find out who is still the team's biggest loser. Thankfully, Masha found somewhere a revolver with a rotating cylinder of n bullet slots able to contain exactly k bullets, now the boys have a chance to resolve the problem once and for all. 

Sasha selects any k out of n slots he wishes and puts bullets there. Roma spins the cylinder so that every of n possible cylinder's shifts is equiprobable. Then the game starts, the players take turns, Sasha starts: he puts the gun to his head and shoots. If there was no bullet in front of the trigger, the cylinder shifts by one position and the weapon is given to Roma for make the same move. The game continues until someone is shot, the survivor is the winner. 

Sasha does not want to lose, so he must choose slots for bullets in such a way as to minimize the probability of its own loss. Of all the possible variant he wants to select the lexicographically minimal one, where an empty slot is lexicographically less than a charged one. 

More formally, the cylinder of n bullet slots able to contain k bullets can be represented as a string of n characters. Exactly k of them are \"X\" (charged slots) and the others are \".\" (uncharged slots). 

Let us describe the process of a shot. Suppose that the trigger is in front of the first character of the string (the first slot). If a shot doesn't kill anyone and the cylinder shifts, then the string shifts left. So the first character becomes the last one, the second character becomes the first one, and so on. But the trigger doesn't move. It will be in front of the first character of the resulting string.

Among all the strings that give the minimal probability of loss, Sasha choose the lexicographically minimal one. According to this very string, he charges the gun. You have to help Sasha to charge the gun. For that, each xi query must be answered: is there a bullet in the positions xi?

Input

The first line contains three integers n, k and p (1 ≤ n ≤ 1018, 0 ≤ k ≤ n, 1 ≤ p ≤ 1000) — the number of slots in the cylinder, the number of bullets and the number of queries. Then follow p lines; they are the queries. Each line contains one integer xi (1 ≤ xi ≤ n) the number of slot to describe.

Please do not use the %lld specificator to read or write 64-bit numbers in С++. It is preferred to use cin, cout streams or the %I64d specificator.

Output

For each query print \".\" if the slot should be empty and \"X\" if the slot should be charged.

Examples

Input

3 1 3
1
2
3


Output

..X

Input

6 3 6
1
2
3
4
5
6


Output

.X.X.X

Input

5 2 5
1
2
3
4
5


Output

...XX

Note

The lexicographical comparison of is performed by the < operator in modern programming languages. The a string is lexicographically less that the b string, if there exists such i (1 ≤ i ≤ n), that ai < bi, and for any j (1 ≤ j < i) aj = bj.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, k, p = map(int, raw_input().split())
s = \"\"
if n % 2 == 1 and k <= n / 2:
    for i in xrange(p):
        x = int(raw_input())
        if x >= n - (k - 1) * 2 and (x % 2 == 0 or x == n):
            s += \"X\"
        else:
            s += \".\"
elif n % 2 == 0:
    for i in xrange(p):
        x = int(raw_input())
        if (n - x) % 2 == 0 and x >= n - 2 * (k - 1):
            s += \"X\"
        elif k > (n + 1) / 2 and x >= (n + 1) - (k - (n + 1) / 2) * 2:
            s += \"X\"
        else:
            s += \".\"
else:
    for i in xrange(p):
        x = int(raw_input())
        if x == n or x % 2 == 0:
            s += \"X\"
        elif x >= n - (k - (n + 1) / 2) * 2:
            s += \"X\"
        else:
            s += \".\"
print s
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Drussianroulettebootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=20, p_min=1, p_max=5, **kwargs):
        self.n_min = n_min
        self.n_max = n_max
        self.p_min = p_min
        self.p_max = p_max
        self.params = locals()  # Capture initialization parameters

    def case_generator(self):
        # Generate n with parity control
        n = random.choice([
            random.randint(self.n_min, self.n_max),
            random.choice([5,7,9]),  # force odd
            random.choice([6,8,10])  # force even
        ])
        
        # Generate k covering critical thresholds
        threshold = (n//2) if n%2 else (n//2)
        k_options = [
            0,
            threshold,
            threshold + 1,
            n
        ]
        k = random.choice(k_options)
        
        # Generate unique queries
        p = random.randint(self.p_min, min(self.p_max, n))
        queries = random.sample(range(1, n+1), p)
        
        return {
            'n': n,
            'k': k,
            'p': p,
            'queries': queries,
            'answers': self.generate_answers(n, k, queries)
        }

    @staticmethod
    def generate_answers(n, k, queries):
        # Generate optimal configuration
        config = ['.']*n
        
        if k > 0:
            if n % 2:  # Odd case
                if k <= (n//2):
                    # Place at largest even positions and last
                    positions = sorted({n - 2*i for i in range(k)} | {n}, reverse=True)[:k]
                else:
                    # Fill trailing positions
                    positions = range(n-k+1, n+1)
            else:      # Even case
                if k <= (n//2):
                    positions = [n - 2*i for i in range(k)]
                else:
                    positions = range(n-k+1, n+1)
            
            # Apply positions with lex order
            for pos in sorted(positions):
                if 1 <= pos <= n:
                    config[pos-1] = 'X'
        
        # Answer queries
        return ''.join([config[x-1] for x in queries])

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""Configure a {question_case['n']}-slot revolver with {question_case['k']} bullets to minimize Sasha's loss probability. Positions: {question_case['queries']}
Answer format: 
[answer]
{''.join(['X' if random.random()>0.5 else '.' for _ in range(question_case['p'])])}
[/answer]"""

    @staticmethod
    def extract_output(output):
        match = re.search(r'\[answer\]\s*([X.]+)\s*\[/answer\]', output, re.IGNORECASE)
        return match.group(1).upper().replace('O', '.') if match else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == identity['answers'] and len(solution) == identity['p']
        except:
            return False
