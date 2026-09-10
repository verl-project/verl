"""# 

### 谜题描述
At a geometry lesson Gerald was given a task: to get vector B out of vector A. Besides, the teacher permitted him to perform the following operations with vector А:

  * Turn the vector by 90 degrees clockwise.
  * Add to the vector a certain vector C.



Operations could be performed in any order any number of times.

Can Gerald cope with the task?

Input

The first line contains integers x1 и y1 — the coordinates of the vector A ( - 108 ≤ x1, y1 ≤ 108). The second and the third line contain in the similar manner vectors B and C (their coordinates are integers; their absolute value does not exceed 108).

Output

Print \"YES\" (without the quotes) if it is possible to get vector B using the given operations. Otherwise print \"NO\" (without the quotes).

Examples

Input

0 0
1 1
0 1


Output

YES


Input

0 0
1 1
1 1


Output

YES


Input

0 0
1 1
2 2


Output

NO

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int p, q;
int x, y;
bool possible(int x, int y) {
  long long bm = 1LL * p * p + 1LL * q * q;
  if (bm == 0) {
    if ((x || y))
      return false;
    else
      return true;
  } else if ((((-1LL * p * x - 1LL * q * y) % bm) ||
              ((-1LL * q * x + 1LL * p * y) % bm)))
    return false;
  else
    return true;
}
int main() {
  bool ans = false;
  int ax, ay;
  cin >> ax >> ay >> x >> y >> p >> q;
  int i;
  int tmp;
  for (i = 0; i < 4; ++i) {
    x -= ax;
    y -= ay;
    ans = ans || (possible(x, y) || possible(-y, x));
    x += ax;
    y += ay;
    tmp = ax;
    ax = ay;
    ay = -tmp;
  }
  if (ans)
    cout << \"YES\";
  else
    cout << \"NO\";
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import Dict, List

def rotate_clockwise(x: int, y: int, times: int) -> (int, int):
    """顺时针旋转向量，times为旋转次数"""
    for _ in range(times % 4):
        x, y = y, -x
    return x, y

def possible(dx: int, dy: int, p: int, q: int) -> bool:
    """验证差分向量是否符合线性组合条件"""
    bm = p**2 + q**2
    if bm == 0:
        return dx == 0 and dy == 0
    return ((-p*dx - q*dy) % bm == 0) and ((-q*dx + p*dy) % bm == 0)

def is_possible(ax: int, ay: int, bx: int, by: int, p: int, q: int) -> bool:
    """验证所有旋转可能性"""
    for rot in range(4):
        rx, ry = rotate_clockwise(ax, ay, rot)
        dx, dy = bx - rx, by - ry
        if possible(dx, dy, p, q) or possible(-dy, dx, p, q):
            return True
    return False

class Evectorsbootcamp(Basebootcamp):
    def __init__(self, max_coordinate: int = 10**3):
        self.max_coord = max_coordinate

    def _gen_solvable_case(self) -> Dict:
        """生成保证可解的案例"""
        ax = random.randint(-self.max_coord, self.max_coord)
        ay = random.randint(-self.max_coord, self.max_coord)
        p = random.randint(-self.max_coord, self.max_coord)
        q = random.randint(-self.max_coord, self.max_coord)
        
        # 随机选择旋转次数和系数
        rot = random.randint(0, 3)
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
        
        # 构造合法B向量
        rx, ry = rotate_clockwise(ax, ay, rot)
        bx = rx + a*p + b*q
        by = ry + a*q - b*p
        return {'A': [ax, ay], 'B': [bx, by], 'C': [p, q]}

    def _gen_unsolvable_zeroC(self) -> Dict:
        """生成C=0时的不可解案例"""
        ax = random.randint(-self.max_coord, self.max_coord)
        ay = random.randint(-self.max_coord, self.max_coord)
        p = q = 0
        
        # 寻找不在旋转对称点上的B
        while True:
            bx = random.randint(-self.max_coord, self.max_coord)
            by = random.randint(-self.max_coord, self.max_coord)
            if not any((bx, by) == rotate_clockwise(ax, ay, r) for r in range(4)):
                return {'A': [ax, ay], 'B': [bx, by], 'C': [p, q]}

    def _gen_unsolvable_general(self) -> Dict:
        """生成普通不可解案例"""
        for _ in range(100):
            case = self._gen_solvable_case()
            ax, ay = case['A']
            bx, by = case['B']
            p, q = case['C']
            
            # 微调B向量破坏可解性
            delta = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
            new_bx = bx + delta[0]
            new_by = by + delta[1]
            if not is_possible(ax, ay, new_bx, new_by, p, q):
                return {'A': [ax, ay], 'B': [new_bx, new_by], 'C': [p, q]}
        return {'A': [0,0], 'B': [1,0], 'C': [0,0]}  # 最终后备案例

    def case_generator(self) -> Dict:
        generators = [
            self._gen_solvable_case,
            self._gen_unsolvable_zeroC,
            self._gen_unsolvable_general
        ]
        return random.choice(generators)()

    @staticmethod
    def prompt_func(case: Dict) -> str:
        a = case['A']
        b = case['B']
        c = case['C']
        return f"""给定初始向量A({a[0]}, {a[1]})，目标向量B({b[0]}, {b[1]})，操作向量C({c[0]}, {c[1]})。允许的操作：
1. 顺时针旋转90度（次数不限）
2. 累加向量C（次数不限）
请判断是否可以转换，并将最终答案用[answer]标签包裹，例如：[answer]YES[/answer]"""

    @staticmethod
    def extract_output(output: str) -> str:
        matches = re.findall(r'\[answer\]\s*(YES|NO)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1].upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution: str, identity: Dict) -> bool:
        ax, ay = identity['A']
        bx, by = identity['B']
        p, q = identity['C']
        expected = 'YES' if is_possible(ax, ay, bx, by, p, q) else 'NO'
        return solution.upper() == expected
