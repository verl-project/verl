"""# 

### 谜题描述
Ivan places knights on infinite chessboard. Initially there are n knights. If there is free cell which is under attack of at least 4 knights then he places new knight in this cell. Ivan repeats this until there are no such free cells. One can prove that this process is finite. One can also prove that position in the end does not depend on the order in which new knights are placed.

Ivan asked you to find initial placement of exactly n knights such that in the end there will be at least ⌊ \frac{n^{2}}{10} ⌋ knights.

Input

The only line of input contains one integer n (1 ≤ n ≤ 10^{3}) — number of knights in the initial placement.

Output

Print n lines. Each line should contain 2 numbers x_{i} and y_{i} (-10^{9} ≤ x_{i},    y_{i} ≤ 10^{9}) — coordinates of i-th knight. For all i ≠ j, (x_{i},    y_{i}) ≠ (x_{j},    y_{j}) should hold. In other words, all knights should be in different cells.

It is guaranteed that the solution exists.

Examples

Input

4


Output

1 1
3 1
1 5
4 4


Input

7


Output

2 1
1 2
4 1
5 2
2 6
5 7
6 6

Note

Let's look at second example:

<image>

Green zeroes are initial knights. Cell (3,    3) is under attack of 4 knights in cells (1,    2), (2,    1), (4,    1) and (5,    2), therefore Ivan will place a knight in this cell. Cell (4,    5) is initially attacked by only 3 knights in cells (2,    6), (5,    7) and (6,    6). But new knight in cell (3,    3) also attacks cell (4,    5), now it is attacked by 4 knights and Ivan will place another knight in this cell. There are no more free cells which are attacked by 4 or more knights, so the process stops. There are 9 knights in the end, which is not less than ⌊ \frac{7^{2}}{10} ⌋ = 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()

for i in range(n):
  if i % 3 == 2:
    print i / 3 * 2 + 1, 3
  else:
    print i / 3 * 2 + i % 3, 0
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict, deque
from bootcamp import Basebootcamp

def simulate_knights_placement(initial_knights):
    current_knights = set()
    coords = [tuple(knight) for knight in initial_knights]
    if len(coords) != len(set(coords)):
        return 0
    for x, y in coords:
        if not (-1e9 <= x <= 1e9 and -1e9 <= y <= 1e9):
            return 0
    current_knights = set(coords)
    knight_moves = [(1,2), (1,-2), (-1,2), (-1,-2), (2,1), (2,-1), (-2,1), (-2,-1)]
    attack_counts = defaultdict(int)
    for x, y in current_knights:
        for dx, dy in knight_moves:
            neighbor = (x + dx, y + dy)
            attack_counts[neighbor] += 1
    queue = deque()
    in_queue = set()
    for cell in attack_counts:
        if attack_counts[cell] >=4 and cell not in current_knights:
            queue.append(cell)
            in_queue.add(cell)
    while queue:
        cell = queue.popleft()
        in_queue.discard(cell)
        if cell in current_knights:
            continue
        if attack_counts[cell] <4:
            continue
        current_knights.add(cell)
        for dx, dy in knight_moves:
            neighbor = (cell[0] + dx, cell[1] + dy)
            attack_counts[neighbor] += 1
            if attack_counts[neighbor] >=4 and neighbor not in current_knights and neighbor not in in_queue:
                queue.append(neighbor)
                in_queue.add(neighbor)
    return len(current_knights)

class Cknightsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=1000, **kwargs):
        super().__init__(**kwargs)
        if not (1 <= min_n <= max_n <= 1000):
            raise ValueError("n must satisfy 1 ≤ min_n ≤ max_n ≤ 1000")
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        return {'n': n}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        required = (n ** 2) // 10
        prompt = f"""You are solving a knight placement puzzle. The task is to arrange {n} knights on an infinite chessboard such that after Ivan's process of adding new knights, the total number becomes at least {required}.

Rules:
1. Initially, place exactly {n} knights on distinct cells.
2. Ivan repeatedly adds a knight to any free cell attacked by at least 4 existing knights until no such cells exist.
3. The final number of knights must be ≥ {required}.

Output:
Provide {n} unique coordinate pairs (x_i, y_i), each between -1e9 and 1e9. Format your answer inside [answer] and [/answer] tags.

Example Format for n=4:
[answer]
1 1
3 1
1 5
4 4
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        coords = []
        for line in last_match.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x = int(parts[0])
                y = int(parts[1])
                coords.append((x, y))
            except ValueError:
                continue
        return coords if coords else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or not isinstance(solution, list):
            return False
        n = identity['n']
        if len(solution) != n:
            return False
        if len(set(solution)) != n:
            return False
        for x, y in solution:
            if not (-1e9 <= x <= 1e9 and -1e9 <= y <= 1e9):
                return False
        required = (n ** 2) // 10
        final_count = simulate_knights_placement(solution)
        return final_count >= required
