"""# 

### 谜题描述
Rumors say that one of Kamal-ol-molk's paintings has been altered. A rectangular brush has been moved right and down on the painting.

Consider the painting as a n × m rectangular grid. At the beginning an x × y rectangular brush is placed somewhere in the frame, with edges parallel to the frame, (1 ≤ x ≤ n, 1 ≤ y ≤ m). Then the brush is moved several times. Each time the brush is moved one unit right or down. The brush has been strictly inside the frame during the painting. The brush alters every cell it has covered at some moment.

You have found one of the old Kamal-ol-molk's paintings. You want to know if it's possible that it has been altered in described manner. If yes, you also want to know minimum possible area of the brush. 

Input

The first line of input contains two integers n and m, (1 ≤ n, m ≤ 1000), denoting the height and width of the painting.

The next n lines contain the painting. Each line has m characters. Character 'X' denotes an altered cell, otherwise it's showed by '.'. There will be at least one altered cell in the painting.

Output

Print the minimum area of the brush in a line, if the painting is possibly altered, otherwise print  - 1.

Examples

Input

4 4
XX..
XX..
XXXX
XXXX


Output

4


Input

4 4
....
.XXX
.XXX
....


Output

2


Input

4 5
XXXX.
XXXX.
.XX..
.XX..


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int getint() {
  int s = 0, o = 1;
  char c;
  for (c = getchar(); c < '0' || c > '9'; c = getchar())
    if (c == '-') o = -1;
  for (; c >= '0' && c <= '9'; c = getchar()) s *= 10, s += c - '0';
  return s * o;
}
const int maxn = 1010;
int a[maxn][maxn], b[maxn][maxn], n, m, len;
inline bool all(int x, int y, int lx, int ly) {
  for (int i = x; i < x + lx; ++i)
    for (int j = y; j < y + ly; ++j)
      if (!a[i][j]) return 0;
  return 1;
}
inline void paint(int x, int y, int lx, int ly) {
  for (int i = x; i < x + lx; ++i)
    for (int j = y; j < y + ly; ++j) b[i][j] = 1;
}
inline int chk(int x, int y, int lx, int ly) {
  if (!all(x, y, lx, ly)) return 2;
  memset(b, 0, sizeof(b));
  paint(x, y, lx, ly);
  for (int t = 0;;) {
    int can[2];
    can[0] = (y + ly < m) & all(x, y + ly, lx, 1);
    can[1] = (x + lx < n) & all(x + lx, y, 1, ly);
    if (!can[0] && !can[1]) break;
    if (can[t]) {
      if (t == 0) {
        for (int i = 0; i < x; i++)
          if (a[i][y + ly] == 1) return 0;
        paint(x, y + ly, lx, 1);
        ++y;
      } else {
        for (int i = 0; i < y; i++)
          if (a[x + lx][i] == 1) return 0;
        paint(x + lx, y, 1, ly);
        ++x;
      }
      continue;
    }
    if (can[t ^ 1]) {
      t ^= 1;
      if (t == 0) {
        for (int i = 0; i < x; i++)
          if (a[i][y + ly] == 1) return 0;
        paint(x, y + ly, lx, 1);
        ++y;
      } else {
        for (int i = 0; i < y; i++)
          if (a[x + lx][i] == 1) return 0;
        paint(x + lx, y, 1, ly);
        ++x;
      }
    }
  }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      if (a[i][j] != b[i][j]) return 2;
  return 1;
}
int work() {
  int x, y, lenx, l, r;
  int fg = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++)
      if (a[i][j]) {
        x = i, y = j, fg = 1;
        break;
      }
    if (fg) break;
  }
  lenx = 1;
  while (x + lenx < n && a[x + lenx][y] == 1) ++lenx;
  l = 0, r = 1;
  while (y + r < m && a[x][y + r] == 1) ++r;
  ++r;
  while (l + 1 < r) {
    int mid = (l + r) >> 1;
    if (chk(x, y, lenx, mid) == 0)
      l = mid;
    else
      r = mid;
  }
  if (chk(x, y, lenx, r) == 2)
    return n * m * 2;
  else
    return lenx * r;
}
char s[1001];
int main(int argc, char const *argv[]) {
  n = getint(), m = getint(), len = max(n, m);
  for (int i = 0; i < n; i++) {
    scanf(\"%s\", s);
    for (int j = 0; j < m; j++) a[i][j] = s[j] == 'X';
  }
  int r = work();
  for (int i = 0; i < len; i++)
    for (int j = 0; j < len; j++)
      if (i < j) swap(a[i][j], a[j][i]);
  swap(n, m);
  r = min(r, work());
  printf(\"%d\n\", r > n * m ? -1 : r);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def solve(n, m, grid):
    a = [[1 if cell == 'X' else 0 for cell in row] for row in grid]
    original_n, original_m = n, m

    def work(a, n, m):
        found = False
        x = y = 0
        for i in range(n):
            for j in range(m):
                if a[i][j]:
                    x, y = i, j
                    found = True
                    break
            if found:
                break
        if not found:
            return n * m * 2  # 无效情况

        lenx = 1
        while x + lenx < n and a[x + lenx][y]:
            lenx += 1

        l = 0
        r = 1
        while y + r < m and a[x][y + r]:
            r += 1
        r += 1  # 初始右边界

        def all_cells(x_check, y_check, lx_check, ly_check):
            if x_check < 0 or y_check < 0 or x_check + lx_check > n or y_check + ly_check > m:
                return False
            for i in range(x_check, x_check + lx_check):
                for j in range(y_check, y_check + ly_check):
                    if not a[i][j]:
                        return False
            return True

        def chk(lx_brush, ly_brush):
            if not all_cells(x, y, lx_brush, ly_brush):
                return 2
            b = [[0] * m for _ in range(n)]
            for i in range(x, x + lx_brush):
                for j in range(y, y + ly_brush):
                    b[i][j] = 1

            current_x, current_y = x, y
            t = 0  # 移动方向标记，0右优先，1下优先
            while True:
                can_right = False
                if current_y + ly_brush < m:
                    can_right = all_cells(current_x, current_y + ly_brush, lx_brush, 1)
                can_down = False
                if current_x + lx_brush < n:
                    can_down = all_cells(current_x + lx_brush, current_y, 1, ly_brush)
                
                if not can_right and not can_down:
                    break

                moved = False
                if can_right and (t == 0 or (not can_down and t == 1)):
                    valid = True
                    for i in range(current_x):
                        if a[i][current_y + ly_brush]:
                            valid = False
                            break
                    if valid:
                        for i in range(current_x, current_x + lx_brush):
                            b[i][current_y + ly_brush] = 1
                        current_y += 1
                        moved = True
                        t = 0
                    else:
                        return 0  # 无效移动路径

                if not moved and can_down and (t == 1 or (not can_right and t == 0)):
                    valid = True
                    for j in range(current_y):
                        if a[current_x + lx_brush][j]:
                            valid = False
                            break
                    if valid:
                        for j in range(current_y, current_y + ly_brush):
                            b[current_x + lx_brush][j] = 1
                        current_x += 1
                        moved = True
                        t = 1
                    else:
                        return 0  # 无效移动路径

                if not moved:
                    break  # 无法移动

            for i in range(n):
                for j in range(m):
                    if a[i][j] != b[i][j]:
                        return 2
            return 1

        left, right = 1, r
        answer = n * m * 2
        while left <= right:
            mid = (left + right) // 2
            res = chk(lenx, mid)
            if res == 1:
                answer = lenx * mid
                right = mid - 1
            elif res == 0:  # 路径无效，需要扩大ly
                left = mid + 1
            else:  # 覆盖不全，需要扩大ly
                left = mid + 1
        return answer if answer <= n * m else n * m * 2

    res1 = work(a, n, m)
    # 转置处理列优先的情况
    a_transposed = [list(row) for row in zip(*a)]
    res2 = work(a_transposed, m, n)
    min_res = min(res1, res2)
    return min_res if min_res <= max(n, m) * max(n, m) else -1

class Ckamalolmolkspaintingbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=5, min_m=2, max_m=5):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m

    def case_generator(self):
        while True:
            n = random.randint(self.min_n, self.max_n)
            m = random.randint(self.min_m, self.max_m)
            grid = []
            x_count = 0
            for _ in range(n):
                row = []
                for _ in range(m):
                    if random.random() < 0.3:
                        row.append('X')
                        x_count += 1
                    else:
                        row.append('.')
                if x_count == 0:
                    row[random.randint(0, m-1)] = 'X'
                    x_count += 1
                grid.append(''.join(row))
            answer = solve(n, m, grid)
            if answer != -1:
                return {
                    'n': n,
                    'm': m,
                    'grid': grid,
                    'answer': answer
                }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        grid = question_case['grid']
        grid_str = '\n'.join(grid)
        return f"""Determine if Kamal-ol-molk's painting could have been altered by a rectangular brush moving strictly right/down. Find the minimal brush area or -1.

Grid ({n}x{m}):
{grid_str}

Rules:
1. Brush starts inside, moves only right/down.
2. All touched cells become 'X'.
3. Answer format: [answer]number[/answer], e.g. [answer]4[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if matches:
            try:
                return int(matches[-1].strip())
            except:
                return None
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
