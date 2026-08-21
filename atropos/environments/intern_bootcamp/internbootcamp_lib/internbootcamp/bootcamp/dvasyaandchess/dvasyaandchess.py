"""# 

### 谜题描述
Vasya decided to learn to play chess. Classic chess doesn't seem interesting to him, so he plays his own sort of chess.

The queen is the piece that captures all squares on its vertical, horizontal and diagonal lines. If the cell is located on the same vertical, horizontal or diagonal line with queen, and the cell contains a piece of the enemy color, the queen is able to move to this square. After that the enemy's piece is removed from the board. The queen cannot move to a cell containing an enemy piece if there is some other piece between it and the queen. 

There is an n × n chessboard. We'll denote a cell on the intersection of the r-th row and c-th column as (r, c). The square (1, 1) contains the white queen and the square (1, n) contains the black queen. All other squares contain green pawns that don't belong to anyone.

The players move in turns. The player that moves first plays for the white queen, his opponent plays for the black queen.

On each move the player has to capture some piece with his queen (that is, move to a square that contains either a green pawn or the enemy queen). The player loses if either he cannot capture any piece during his move or the opponent took his queen during the previous move. 

Help Vasya determine who wins if both players play with an optimal strategy on the board n × n.

Input

The input contains a single number n (2 ≤ n ≤ 109) — the size of the board.

Output

On the first line print the answer to problem — string \"white\" or string \"black\", depending on who wins if the both players play optimally. 

If the answer is \"white\", then you should also print two integers r and c representing the cell (r, c), where the first player should make his first move to win. If there are multiple such cells, print the one with the minimum r. If there are still multiple squares, print the one with the minimum c.

Examples

Input

2


Output

white
1 2


Input

3


Output

black

Note

In the first sample test the white queen can capture the black queen at the first move, so the white player wins.

In the second test from the statement if the white queen captures the green pawn located on the central vertical line, then it will be captured by the black queen during the next move. So the only move for the white player is to capture the green pawn located at (2, 1). 

Similarly, the black queen doesn't have any other options but to capture the green pawn located at (2, 3), otherwise if it goes to the middle vertical line, it will be captured by the white queen.

During the next move the same thing happens — neither the white, nor the black queen has other options rather than to capture green pawns situated above them. Thus, the white queen ends up on square (3, 1), and the black queen ends up on square (3, 3). 

In this situation the white queen has to capture any of the green pawns located on the middle vertical line, after that it will be captured by the black queen. Thus, the player who plays for the black queen wins.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
print \"white\n1 2\" if int(raw_input()) % 2 is 0 else \"black\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Dvasyaandchessbootcamp(Basebootcamp):
    def __init__(self, n=None, min_n=2, max_n=10**9, parity=None):
        self.n = n
        self.min_n = max(2, min_n)
        self.max_n = min(max_n, 10**9)
        self.parity = parity
        
        # Validate parameters on initialization
        if self.n is not None:
            if not (self.min_n <= self.n <= self.max_n):
                raise ValueError(f"n must be between {self.min_n} and {self.max_n}")
            if self.parity:
                actual_parity = 'even' if self.n % 2 == 0 else 'odd'
                if self.parity != actual_parity:
                    raise ValueError(f"Specified parity {self.parity} conflicts with fixed n={self.n}")

    def case_generator(self):
        if self.n is not None:
            return {'n': self.n}
        
        # 高效生成候选值算法
        def generate_candidate():
            if self.parity == 'even':
                start = self.min_n if self.min_n % 2 == 0 else self.min_n + 1
                end = self.max_n if self.max_n % 2 == 0 else self.max_n - 1
                if start > end:
                    raise ValueError("No even numbers in range")
                return random.randrange(start, end + 1, 2)
            
            elif self.parity == 'odd':
                start = self.min_n if self.min_n % 2 == 1 else self.min_n + 1
                end = self.max_n if self.max_n % 2 == 1 else self.max_n - 1
                if start > end:
                    raise ValueError("No odd numbers in range")
                return random.randrange(start, end + 1, 2)
            
            else:  # 无奇偶性要求
                return random.randint(self.min_n, self.max_n)
        
        try:
            n = generate_candidate()
            return {'n': n}
        except ValueError as e:
            raise ValueError(f"No valid n in range [{self.min_n}, {self.max_n}] with parity {self.parity}") from e
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        return f"""Vasya is playing a variant of chess on a {n}x{n} board. The white queen starts at (1,1), and the black queen at (1,{n}). All other cells contain green pawns. Players alternate turns, with white moving first. Each move must capture an enemy piece (green pawn or opposing queen) along a clear path. The player loses if unable to capture or if their queen was captured.

Given n = {n}, determine the winner with optimal play. If white wins, specify their first move's coordinates (smallest row then column).

Output your answer within [answer] tags. Examples:
[answer]white
1 2[/answer]
[answer]black[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        solution = solution.strip()
        lines = [line.strip() for line in solution.split('\n') if line.strip()]

        if n % 2 == 0:
            return (len(lines) == 2 and 
                    lines[0].lower() == 'white' and 
                    list(map(int, lines[1].split())) == [1, 2])
        else:
            return len(lines) == 1 and lines[0].lower() == 'black'
