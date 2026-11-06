"""# 

### 谜题描述
Old MacDonald has a farm and a large potato field, (1010 + 1) × (1010 + 1) square meters in size. The field is divided into square garden beds, each bed takes up one square meter.

Old McDonald knows that the Colorado potato beetle is about to invade his farm and can destroy the entire harvest. To fight the insects, Old McDonald wants to spray some beds with insecticides.

So Old McDonald went to the field, stood at the center of the central field bed and sprayed this bed with insecticides. Now he's going to make a series of movements and spray a few more beds. During each movement Old McDonald moves left, right, up or down the field some integer number of meters. As Old McDonald moves, he sprays all the beds he steps on. In other words, the beds that have any intersection at all with Old McDonald's trajectory, are sprayed with insecticides.

When Old McDonald finished spraying, he wrote out all his movements on a piece of paper. Now he wants to know how many beds won't be infected after the invasion of the Colorado beetles.

It is known that the invasion of the Colorado beetles goes as follows. First some bed on the field border gets infected. Than any bed that hasn't been infected, hasn't been sprayed with insecticides and has a common side with an infected bed, gets infected as well. Help Old McDonald and determine the number of beds that won't be infected by the Colorado potato beetle.

Input

The first line contains an integer n (1 ≤ n ≤ 1000) — the number of Old McDonald's movements.

Next n lines contain the description of Old McDonald's movements. The i-th of these lines describes the i-th movement. Each movement is given in the format \"di xi\", where di is the character that determines the direction of the movement (\"L\", \"R\", \"U\" or \"D\" for directions \"left\", \"right\", \"up\" and \"down\", correspondingly), and xi (1 ≤ xi ≤ 106) is an integer that determines the number of meters in the movement.

Output

Print a single integer — the number of beds that won't be infected by the Colorado potato beetle.

Please do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5
R 8
U 9
L 9
D 8
L 2


Output

101

Input

7
R 10
D 2
L 7
U 9
D 2
R 3
D 10


Output

52

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
struct Rectangle {
  int X0, X1, Y0, Y1;
};
int Get() {
  char c;
  while (c = getchar(), c < '0' || c > '9')
    ;
  int X = 0;
  while (c >= '0' && c <= '9') {
    X = X * 10 + c - 48;
    c = getchar();
  }
  return X;
}
char GetDirection() {
  char c;
  while (c = getchar(), c < 'A' || c > 'Z') return c;
}
const int V[4][2] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
int main() {
  int N = Get(), X = 0, Y = 0;
  static Rectangle A[1000];
  static int DataX[2000], DataY[2000];
  for (int i = 0; i < N; i++) {
    int _X = X, _Y = Y;
    char c = GetDirection();
    int Len = Get();
    if (c == 'R') X += Len;
    if (c == 'U') Y += Len;
    if (c == 'L') X -= Len;
    if (c == 'D') Y -= Len;
    A[i] = (Rectangle){min(X, _X), max(X, _X) + 1, min(Y, _Y), max(Y, _Y) + 1};
    DataX[i * 2] = A[i].X0;
    DataX[i * 2 + 1] = A[i].X1;
    DataY[i * 2] = A[i].Y0;
    DataY[i * 2 + 1] = A[i].Y1;
  }
  sort(DataX, DataX + N * 2);
  sort(DataY, DataY + N * 2);
  int R = 1, C = 1;
  for (int i = 1; i < N * 2; i++) {
    if (DataX[i] != DataX[i - 1]) DataX[R++] = DataX[i];
    if (DataY[i] != DataY[i - 1]) DataY[C++] = DataY[i];
  }
  static bool Map[2000][2000];
  for (int i = 0; i < N; i++) {
    int P0 = 0;
    while (DataX[P0] < A[i].X0) P0++;
    int P1 = P0;
    while (DataX[P1] < A[i].X1) P1++;
    int Q0 = 0;
    while (DataY[Q0] < A[i].Y0) Q0++;
    int Q1 = Q0;
    while (DataY[Q1] < A[i].Y1) Q1++;
    for (int j = P0; j < P1; j++)
      for (int k = Q0; k < Q1; k++) Map[j][k] = true;
  }
  static bool Flag[2000][2000];
  for (int i = 0; i < R; i++)
    for (int j = 0; j < C; j++) {
      if (Map[i][j] || Flag[i][j]) continue;
      int Begin = 0, End = 1;
      static int Qx[4000000], Qy[4000000];
      Qx[0] = i;
      Qy[0] = j;
      Flag[i][j] = true;
      while (Begin < End) {
        int X0 = Qx[Begin], Y0 = Qy[Begin];
        Begin++;
        for (int D = 0; D < 4; D++) {
          int X1 = X0 + V[D][0], Y1 = Y0 + V[D][1];
          if (X1 < 0 || X1 == R || Y1 < 0 || Y1 == C) continue;
          if (Map[X1][Y1] || Flag[X1][Y1]) continue;
          Qx[End] = X1;
          Qy[End] = Y1;
          End++;
          Flag[X1][Y1] = true;
        }
      }
      bool Out = false;
      for (int k = 0; !Out && k < End; k++)
        Out = (!Qx[k]) || (Qx[k] == R - 1) || (!Qy[k]) || (Qy[k] == C - 1);
      if (!Out)
        for (int k = 0; k < End; k++) Map[Qx[k]][Qy[k]] = true;
    }
  long long Ans = 0;
  for (int i = 1; i < R; i++)
    for (int j = 1; j < C; j++)
      if (Map[i - 1][j - 1])
        Ans += (long long)(DataX[i] - DataX[i - 1]) * (DataY[j] - DataY[j - 1]);
  cout << Ans << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import bisect
import random
from bootcamp import Basebootcamp
import re

class Ecoloradopotatobeetlebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 1)
        self.n_max = params.get('n_max', 20)
        self.step_min = params.get('step_min', 1)
        self.step_max = params.get('step_max', 1000)

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        directions = ['L', 'R', 'U', 'D']
        movements = []
        for _ in range(n):
            d = random.choice(directions)
            step = random.randint(self.step_min, self.step_max)
            movements.append((d, step))
        # 修复方法调用方式
        correct_answer = self._calculate_answer(movements)
        return {
            'n': n,
            'movements': movements,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        movements = question_case['movements']
        n = question_case['n']
        input_lines = [str(n)] + [f"{d} {xi}" for d, xi in movements]
        input_str = '\n'.join(input_lines)
        problem = (
            "Old MacDonald has a (1010+1)×(1010+1) square meter field divided into 1x1 beds. He starts at the center, sprays it, then moves/direction steps, spraying all beds along his path. Beetles infect border beds first, spreading to adjacent unsprayed beds. Calculate the number of uninfected beds.\n"
            "Input:\n"
            f"{input_str}\n"
            "Output the answer inside [answer] and [/answer], e.g., [answer]101[/answer].\n"
        )
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']

    @staticmethod
    def _calculate_answer(movements):
        X, Y = 0, 0
        A = []
        field_size = 1010  # 1011x1011 field (0..1010 inclusive)
        for d, xi in movements:
            old_X, old_Y = X, Y
            if d == 'R':
                X += xi
            elif d == 'L':
                X -= xi
            elif d == 'U':
                Y += xi
            elif d == 'D':
                Y -= xi
            x0 = min(X, old_X)
            x1 = max(X, old_X) + 1
            y0 = min(Y, old_Y)
            y1 = max(Y, old_Y) + 1
            A.append((x0, x1, y0, y1))

        # 离散化处理
        DataX, DataY = [], []
        for x0, x1, y0, y1 in A:
            DataX.extend([x0, x1])
            DataY.extend([y0, y1])
        # 添加field边界保证离散化覆盖
        DataX.extend([-1, 0, field_size, field_size+1])
        DataY.extend([-1, 0, field_size, field_size+1])
        DataX = sorted(list(set(DataX)))
        DataY = sorted(list(set(DataY)))

        R, C = len(DataX), len(DataY)
        Map = [[False]*C for _ in range(R)]

        # 标记喷洒区域
        for x0_rect, x1_rect, y0_rect, y1_rect in A:
            P0 = bisect.bisect_left(DataX, x0_rect)
            P1 = bisect.bisect_left(DataX, x1_rect)
            Q0 = bisect.bisect_left(DataY, y0_rect)
            Q1 = bisect.bisect_left(DataY, y1_rect)
            for i in range(P0, P1):
                for j in range(Q0, Q1):
                    if i < R and j < C:
                        Map[i][j] = True

        # Flood fill算法修正
        visited = [[False]*C for _ in range(R)]
        total_safe = 0

        for i in range(R):
            for j in range(C):
                if Map[i][j] or visited[i][j]:
                    continue
                
                queue = [(i, j)]
                visited[i][j] = True
                front = 0
                is_safe = True
                has_boundary = False
                
                # 检查区域是否包含field边界
                while front < len(queue):
                    x, y = queue[front]
                    front += 1
                    
                    # 检查该单元格是否覆盖field边界
                    x_start = DataX[x]
                    x_end = DataX[x+1] if x+1 < R else DataX[x]
                    y_start = DataY[y]
                    y_end = DataY[y+1] if y+1 < C else DataY[y]
                    
                    # 判断是否覆盖field的四个边界
                    if (x_start <= 0 < x_end) or (x_start <= field_size < x_end) or \
                       (y_start <= 0 < y_end) or (y_start <= field_size < y_end):
                        has_boundary = True
                        break
                    
                    # 扩展相邻单元格
                    for dx, dy in [(-1,0),(0,-1),(0,1),(1,0)]:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < R and 0 <= ny < C:
                            if not Map[nx][ny] and not visited[nx][ny]:
                                visited[nx][ny] = True
                                queue.append((nx, ny))
                
                if not has_boundary:
                    # 计算安全区域面积
                    for x_cell, y_cell in queue:
                        dx = DataX[x_cell+1] - DataX[x_cell] if x_cell+1 < R else 0
                        dy = DataY[y_cell+1] - DataY[y_cell] if y_cell+1 < C else 0
                        total_safe += dx * dy

        return total_safe
