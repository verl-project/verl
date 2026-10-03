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
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int y = 0, k = 0;
  while (y != n) {
    if (k % 2 == 0) {
      cout << k << \"  0\" << endl;
      y++;
    } else {
      if (n - y == 1) {
        cout << k << \"  0\" << endl;
        y++;
      } else if (k % 2 == 1) {
        cout << k << \"  0\" << endl;
        cout << k << \"  3\" << endl;
        y += 2;
      }
    }
    k++;
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random

class Fknightsbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.n = params.get('n', 4)  # 默认n=4

    def case_generator(self):
        n = random.randint(1, 1000)  # 随机生成n，用于测试
        positions = self.generate_initial_positions(n)
        return {'n': n, 'positions': positions}

    @staticmethod
    def generate_initial_positions(n):
        positions = []
        y = 0
        k = 0
        while y < n:
            if k % 2 == 0:
                positions.append((k, 0))
                y += 1
            else:
                if n - y == 1:
                    positions.append((k, 0))
                    y += 1
                else:
                    positions.append((k, 0))
                    positions.append((k, 3))
                    y += 2
            k += 1
        return positions

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        required_knights = (n ** 2) // 10
        prompt = f"Given n = {n}, find an initial placement of exactly {n} knights on an infinite chessboard such that after Ivan's process, the total number of knights is at least {required_knights}. Please output the coordinates of the knights, each on a separate line, using the format 'x y'. Make sure all knights are placed on distinct cells. Place your answer within [answer] tags."
        return prompt

    @staticmethod
    def extract_output(output):
        pattern = r'\b(-?\d+)\s+(-?\d+)\b'
        matches = re.findall(pattern, output)
        if not matches:
            return None
        solution = []
        for x, y in matches:
            solution.append((int(x), int(y)))
        unique_solution = list({(x, y) for x, y in solution})
        return unique_solution

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or not identity:
            return False
        n = identity['n']
        if len(solution) != n:
            return False
        correct_positions = identity['positions']
        solution_set = set(solution)
        correct_set = set(correct_positions)
        return solution_set == correct_set
