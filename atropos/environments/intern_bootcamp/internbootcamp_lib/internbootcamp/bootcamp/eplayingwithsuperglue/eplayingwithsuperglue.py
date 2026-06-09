"""# 

### 谜题描述
Two players play a game. The game is played on a rectangular board with n × m squares. At the beginning of the game two different squares of the board have two chips. The first player's goal is to shift the chips to the same square. The second player aims to stop the first one with a tube of superglue.

We'll describe the rules of the game in more detail.

The players move in turns. The first player begins.

With every move the first player chooses one of his unglued chips, and shifts it one square to the left, to the right, up or down. It is not allowed to move a chip beyond the board edge. At the beginning of a turn some squares of the board may be covered with a glue. The first player can move the chip to such square, in this case the chip gets tightly glued and cannot move any longer.

At each move the second player selects one of the free squares (which do not contain a chip or a glue) and covers it with superglue. The glue dries long and squares covered with it remain sticky up to the end of the game.

If, after some move of the first player both chips are in the same square, then the first player wins. If the first player cannot make a move (both of his chips are glued), then the second player wins. Note that the situation where the second player cannot make a move is impossible — he can always spread the glue on the square from which the first player has just moved the chip.

We will further clarify the case where both chips are glued and are in the same square. In this case the first player wins as the game ends as soon as both chips are in the same square, and the condition of the loss (the inability to move) does not arise.

You know the board sizes and the positions of the two chips on it. At the beginning of the game all board squares are glue-free. Find out who wins if the players play optimally.

Input

The first line contains six integers n, m, x1, y1, x2, y2 — the board sizes and the coordinates of the first and second chips, correspondingly (1 ≤ n, m ≤ 100; 2 ≤ n × m; 1 ≤ x1, x2 ≤ n; 1 ≤ y1, y2 ≤ m). The numbers in the line are separated by single spaces.

It is guaranteed that the chips are located in different squares.

Output

If the first player wins, print \"First\" without the quotes. Otherwise, print \"Second\" without the quotes.

Examples

Input

1 6 1 2 1 6


Output

First

Input

6 5 4 3 2 1


Output

First

Input

10 10 1 1 10 10


Output

Second

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m, x1, y1, x2, y2 = map(int, raw_input().split())
dx = abs(x1 - x2)
dy = abs(y1 - y2)
if dx < dy:
    t = dx
    dx = dy
    dy = t
if dx <= 3 and dy <= 3 or (dx, dy) in [(4, 0), (4, 1), (4, 2)]:
    print \"First\"
else:
    print \"Second\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Eplayingwithsupergluebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100, min_m=1, max_m=100):
        self.min_n = max(min_n, 1)
        self.max_n = max(self.min_n, min(max_n, 100))  # Ensure max_n ≤ 100
        self.min_m = max(min_m, 1)
        self.max_m = max(self.min_m, min(max_m, 100))  # Ensure max_m ≤ 100
    
    def case_generator(self):
        # Generate valid board size
        while True:
            n = random.randint(self.min_n, self.max_n)
            m = random.randint(self.min_m, self.max_m)
            if n * m >= 2:
                break
        
        # Generate distinct positions
        while True:
            x1, y1 = random.randint(1, n), random.randint(1, m)
            x2, y2 = random.randint(1, n), random.randint(1, m)
            if (x1, y1) != (x2, y2):
                break
        
        return {
            'n': n,
            'm': m,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        prompt = f"""Two players play a game on a {question_case['n']}x{question_case['m']} grid. Chips start at ({question_case['x1']},{question_case['y1']}) and ({question_case['x2']},{question_case['y2']}).

Rules:
1. First player moves one unglued chip (up/down/left/right) each turn
2. Second player places glue on an empty square each turn
3. First wins if chips meet, Second wins if both chips get glued

Determine the winner under optimal play. Put your final answer within [answer] tags, like [answer]First[/answer] or [answer]Second[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # Match last occurrence with flexible tag closing
        matches = re.findall(r'\[answer\](.*?)(?=\s*\[/?answer\]|$)', output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip().lower()
        if answer in ('first', 'second'):
            return answer.capitalize()
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        x1, y1 = identity['x1'], identity['y1']
        x2, y2 = identity['x2'], identity['y2']
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        # Ensure dx >= dy
        if dx < dy:
            dx, dy = dy, dx
        
        # Check winning conditions
        first_wins = (dx <= 3 and dy <= 3) or (dx, dy) in {(4,0), (4,1), (4,2)}
        return (solution == 'First') == first_wins
