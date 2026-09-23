"""# 

### 谜题描述
Igor has been into chess for a long time and now he is sick of the game by the ordinary rules. He is going to think of new rules of the game and become world famous.

Igor's chessboard is a square of size n × n cells. Igor decided that simple rules guarantee success, that's why his game will have only one type of pieces. Besides, all pieces in his game are of the same color. The possible moves of a piece are described by a set of shift vectors. The next passage contains a formal description of available moves.

Let the rows of the board be numbered from top to bottom and the columns be numbered from left to right from 1 to n. Let's assign to each square a pair of integers (x, y) — the number of the corresponding column and row. Each of the possible moves of the piece is defined by a pair of integers (dx, dy); using this move, the piece moves from the field (x, y) to the field (x + dx, y + dy). You can perform the move if the cell (x + dx, y + dy) is within the boundaries of the board and doesn't contain another piece. Pieces that stand on the cells other than (x, y) and (x + dx, y + dy) are not important when considering the possibility of making the given move (for example, like when a knight moves in usual chess).

Igor offers you to find out what moves his chess piece can make. He placed several pieces on the board and for each unoccupied square he told you whether it is attacked by any present piece (i.e. whether some of the pieces on the field can move to that cell). Restore a possible set of shift vectors of the piece, or else determine that Igor has made a mistake and such situation is impossible for any set of shift vectors.

Input

The first line contains a single integer n (1 ≤ n ≤ 50).

The next n lines contain n characters each describing the position offered by Igor. The j-th character of the i-th string can have the following values:

  * o — in this case the field (i, j) is occupied by a piece and the field may or may not be attacked by some other piece;
  * x — in this case field (i, j) is attacked by some piece;
  * . — in this case field (i, j) isn't attacked by any piece.



It is guaranteed that there is at least one piece on the board.

Output

If there is a valid set of moves, in the first line print a single word 'YES' (without the quotes). Next, print the description of the set of moves of a piece in the form of a (2n - 1) × (2n - 1) board, the center of the board has a piece and symbols 'x' mark cells that are attacked by it, in a format similar to the input. See examples of the output for a full understanding of the format. If there are several possible answers, print any of them.

If a valid set of moves does not exist, print a single word 'NO'.

Examples

Input

5
oxxxx
x...x
x...x
x...x
xxxxo


Output

YES
....x....
....x....
....x....
....x....
xxxxoxxxx
....x....
....x....
....x....
....x....


Input

6
.x.x..
x.x.x.
.xo..x
x..ox.
.x.x.x
..x.x.


Output

YES
...........
...........
...........
....x.x....
...x...x...
.....o.....
...x...x...
....x.x....
...........
...........
...........


Input

3
o.x
oxx
o.x


Output

NO

Note

In the first sample test the piece is a usual chess rook, and in the second sample test the piece is a usual chess knight.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
A = [list(raw_input()) for i in xrange(n)]
dxy = [[\"x\"] * (2 * n - 1) for i in xrange(2 * n - 1)]
for y in xrange(n):
    for x in xrange(n):
        if A[y][x] == \"o\":
            for dy in xrange(-y, n - y):
                for dx in xrange(-x, n - x):
                    if dx == dy == 0: continue
                    if A[y + dy][x + dx] in \".\":
                        dxy[dy + n - 1][dx + n - 1] = \".\"

for y in xrange(n):
    for x in xrange(n):
        if A[y][x] == \"o\":
            for dy in xrange(-y, n - y):
                for dx in xrange(-x, n - x):
                    if A[y + dy][x + dx] == \"x\" and dxy[dy + n - 1][dx + n - 1] == \"x\":
                        A[y + dy][x + dx] = \".\"

for y in xrange(n):
    if \"x\" in A[y]:
        print \"NO\"
        exit()
        
dxy[n - 1][n - 1] = \"o\"                     
print \"YES\"
for line in dxy:
    print \"\".join(line)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dweirdchessbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)
        self.min_o = params.get('min_o', 1)
        self.max_o = params.get('max_o', 3)
        self.move_types = params.get('move_types', ['random', 'rook', 'knight'])
        self.current_move_type = params.get('move_type', 'random')

    def case_generator(self):
        n = self.n
        move_vectors = self._generate_move_vectors(n)
        o_positions = self._generate_o_positions(n)
        grid = self._generate_grid(n, o_positions, move_vectors)
        return {
            'n': n,
            'grid': [''.join(row) for row in grid]
        }

    def _generate_move_vectors(self, n):
        if self.current_move_type == 'rook':
            vectors = []
            for dx in range(-n+1, n):
                if dx != 0:
                    vectors.append((dx, 0))
            for dy in range(-n+1, n):
                if dy != 0:
                    vectors.append((0, dy))
            return list(set(vectors))
        elif self.current_move_type == 'knight':
            return [ (dx, dy) for dx in (-2, -1, 1, 2) for dy in (-2, -1, 1, 2) if abs(dx) + abs(dy) == 3 ]
        else:
            vectors = []
            for _ in range(random.randint(3, 6)):
                dx = random.randint(-n+1, n-1)
                dy = random.randint(-n+1, n-1)
                if dx == 0 and dy == 0:
                    continue
                vectors.append((dx, dy))
            return list(set(vectors))

    def _generate_o_positions(self, n):
        count = random.randint(self.min_o, self.max_o)
        positions = set()
        while len(positions) < count:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            positions.add((x, y))
        return list(positions)

    def _generate_grid(self, n, o_positions, move_vectors):
        grid = [['.' for _ in range(n)] for _ in range(n)]
        o_coords = set((x, y) for x, y in o_positions)

        for x, y in o_coords:
            grid[y][x] = 'o'

        for x, y in o_coords:
            for dx, dy in move_vectors:
                tx = x + dx
                ty = y + dy
                if 0 <= tx < n and 0 <= ty < n:
                    if (tx, ty) not in o_coords:
                        grid[ty][tx] = 'x'

        return grid

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        grid = '\n'.join(question_case['grid'])
        return f"""You are participating in Igor's chess puzzle challenge. The chessboard is a {n}x{n} grid where:
- 'o' represents a piece.
- 'x' represents a cell attacked by at least one piece.
- '.' represents a cell not attacked by any piece.

The current board configuration is:
{grid}

Your task is to determine if there exists a valid set of move vectors for the pieces to achieve the attack pattern shown. If possible, output 'YES' followed by the move vectors matrix. The matrix should be a {2*n-1}x{2*n-1} grid with 'o' at the center and 'x' marking valid moves. If impossible, output 'NO'.

Format your answer as follows:
[answer]
YES
....x....
...x.x...
..x...x..
.x.....x.
xxxxoxxxx
.x.....x.
..x...x..
...x.x...
....x....
[/answer]
or
[answer]
NO
[/answer]

Replace the example with your solution. Ensure your answer is enclosed within [answer] and [/answer] tags."""

    @staticmethod
    def extract_output(output):
        pattern = re.compile(r'\[answer\](.*?)\[/answer\]', re.DOTALL)
        matches = pattern.findall(output)
        if not matches:
            return None
        answer_content = matches[-1].strip()
        lines = [line.strip() for line in answer_content.split('\n') if line.strip()]

        if not lines:
            return None
        first_line = lines[0].upper()
        if first_line == 'NO':
            return ['NO']
        elif first_line == 'YES':
            if len(lines) < 2:
                return None
            matrix = lines[1:]
            size = len(matrix)
            if (size + 1) % 2 != 0:
                return None
            n = (size + 1) // 2
            if any(len(row) != size for row in matrix):
                return None
            if matrix[n-1][n-1] != 'o':
                return None
            for row in matrix:
                if not all(c in {'x', '.', 'o'} for c in row):
                    return None
            return matrix
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == ['NO']:
            return False

        n = identity['n']
        input_grid = identity['grid']

        try:
            size = len(solution)
            if size != 2 * n - 1:
                return False
            for row in solution:
                if len(row) != size:
                    return False
            if solution[n-1][n-1] != 'o':
                return False

            move_vectors = []
            for i in range(size):
                for j in range(size):
                    if solution[i][j] == 'x':
                        dx = j - (n-1)
                        dy = i - (n-1)
                        move_vectors.append((dx, dy))

            temp_grid = [list(row) for row in input_grid]
            o_positions = [(x, y) for y in range(n) for x in range(n) if input_grid[y][x] == 'o']

            for x, y in o_positions:
                for dx, dy in move_vectors:
                    tx = x + dx
                    ty = y + dy
                    if 0 <= tx < n and 0 <= ty < n:
                        if temp_grid[ty][tx] == 'x':
                            temp_grid[ty][tx] = '.'

            for row in temp_grid:
                if 'x' in row:
                    return False
            return True
        except:
            return False
