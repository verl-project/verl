"""# 

### 谜题描述
Inna and Dima bought a table of size n × m in the shop. Each cell of the table contains a single letter: \"D\", \"I\", \"M\", \"A\".

Inna loves Dima, so she wants to go through his name as many times as possible as she moves through the table. For that, Inna acts as follows:

  1. initially, Inna chooses some cell of the table where letter \"D\" is written; 
  2. then Inna can move to some side-adjacent table cell that contains letter \"I\"; then from this cell she can go to one of the side-adjacent table cells that contains the written letter \"M\"; then she can go to a side-adjacent cell that contains letter \"A\". Then Inna assumes that she has gone through her sweetheart's name; 
  3. Inna's next move can be going to one of the side-adjacent table cells that contains letter \"D\" and then walk on through name DIMA in the similar manner. Inna never skips a letter. So, from the letter \"D\" she always goes to the letter \"I\", from the letter \"I\" she always goes the to letter \"M\", from the letter \"M\" she always goes to the letter \"A\", and from the letter \"A\" she always goes to the letter \"D\". 



Depending on the choice of the initial table cell, Inna can go through name DIMA either an infinite number of times or some positive finite number of times or she can't go through his name once. Help Inna find out what maximum number of times she can go through name DIMA.

Input

The first line of the input contains two integers n and m (1 ≤ n, m ≤ 103). 

Then follow n lines that describe Inna and Dima's table. Each line contains m characters. Each character is one of the following four characters: \"D\", \"I\", \"M\", \"A\". 

Note that it is not guaranteed that the table contains at least one letter \"D\".

Output

If Inna cannot go through name DIMA once, print on a single line \"Poor Dima!\" without the quotes. If there is the infinite number of names DIMA Inna can go through, print \"Poor Inna!\" without the quotes. Otherwise print a single integer — the maximum number of times Inna can go through name DIMA.

Examples

Input

1 2
DI


Output

Poor Dima!


Input

2 2
MA
ID


Output

Poor Inna!


Input

5 5
DIMAD
DIMAI
DIMAM
DDMAA
AAMID


Output

4

Note

Notes to the samples:

In the first test sample, Inna cannot go through name DIMA a single time.

In the second test sample, Inna can go through the infinite number of words DIMA. For that, she should move in the clockwise direction starting from the lower right corner.

In the third test sample the best strategy is to start from the cell in the upper left corner of the table. Starting from this cell, Inna can go through name DIMA four times. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAX = 1002;
string str[MAX];
int n, m, dp[MAX][MAX], dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, -1, 1},
                        vis[MAX][MAX], cur[MAX][MAX];
char next(char a) {
  if (a == 'D') return 'I';
  if (a == 'I') return 'M';
  if (a == 'M') return 'A';
  return 'D';
}
bool valid(int x, int y) { return x >= 0 && y >= 0 && x < n && y < m; }
void dfs(int x, int y) {
  for (int i = 0; i < 4; i++) {
    int sx = x + dx[i], sy = y + dy[i];
    if (not valid(sx, sy) or str[sx][sy] != next(str[x][y])) continue;
    if (vis[sx][sy]) {
      if (cur[sx][sy]) {
        cout << \"Poor Inna!\" << endl;
        exit(0);
      }
    } else {
      vis[sx][sy] = cur[sx][sy] = 1;
      dfs(sx, sy);
    }
    dp[x][y] = max(dp[x][y], 1 + dp[sx][sy]);
    cur[sx][sy] = 0;
  }
  dp[x][y] = max(dp[x][y], 1);
}
int main() {
  cin >> n >> m;
  for (int i = 0; i < n; i++) cin >> str[i];
  int ans = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) {
      if (str[i][j] == 'D' and not vis[i][j]) {
        vis[i][j] = cur[i][j] = 1, dfs(i, j);
        cur[i][j] = 0;
        ans = max(ans, dp[i][j]);
      }
    }
  if (ans / 4 == 0)
    cout << \"Poor Dima!\" << endl;
  else
    cout << ans / 4 << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import Dict, Any

class Cinnaanddimabootcamp(Basebootcamp):
    def __init__(self, n: int = 5, m: int = 5):
        self.n = n
        self.m = m

    def case_generator(self) -> Dict[str, Any]:
        grid = []
        for _ in range(self.n):
            row = []
            for _ in range(self.m):
                row.append(random.choice(['D', 'I', 'M', 'A']))
            grid.append(''.join(row))
        return {
            'n': self.n,
            'm': self.m,
            'grid': grid
        }

    @staticmethod
    def prompt_func(question_case: Dict[str, Any]) -> str:
        grid = question_case['grid']
        n = question_case['n']
        m = question_case['m']
        prompt = f"给定一个{n}行{m}列的表格，如下所示：\n"
        for row in grid:
            prompt += f"行：{row}\n"
        prompt += "\n"
        prompt += "Inna希望尽可能多地走Cinnaanddima的循环。规则如下：\n"
        prompt += "1. Inna从一个D点开始。\n"
        prompt += "2. 每一步必须移动到相邻的格子（上下左右），且下一个字母必须是下一个字母（D→I→M→A→D）。\n"
        prompt += "3. 计算Inna能走多少次完整的Cinnaanddima循环。\n"
        prompt += "\n"
        prompt += "输出要求：\n"
        prompt += "如果Inna无法完成一次Cinnaanddima循环，输出“Poor Dima!”。\n"
        prompt += "如果存在无限循环的情况，输出“Poor Inna!”。\n"
        prompt += "否则，输出最大次数。\n"
        prompt += "\n"
        prompt += "请将答案放在[answer]标签中，例如：\n"
        prompt += "[answer]4[/answer]\n"
        prompt += "或者\n"
        prompt += "[answer]Poor Dima![/answer]\n"
        prompt += "或者\n"
        prompt += "[answer]Poor Inna![/answer]\n"
        return prompt

    @staticmethod
    def extract_output(output: str) -> str:
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if matches:
            return matches[-1].strip()
        else:
            return None

    @classmethod
    def _verify_correction(cls, solution: str, identity: Dict[str, Any]) -> bool:
        grid = identity['grid']
        n = identity['n']
        m = identity['m']
        correct_ans = cls._compute_correct_answer(grid, n, m)
        return solution == correct_ans

    @classmethod
    def _compute_correct_answer(cls, grid: list, n: int, m: int) -> str:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dp = [[0 for _ in range(m)] for __ in range(n)]
        visited = [[False for _ in range(m)] for __ in range(n)]
        in_cycle = [[False for _ in range(m)] for __ in range(n)]
        has_infinite = False

        def next_char(c: str) -> str:
            if c == 'D':
                return 'I'
            elif c == 'I':
                return 'M'
            elif c == 'M':
                return 'A'
            else:
                return 'D'

        for i in range(n):
            for j in range(m):
                if grid[i][j] == 'D' and not visited[i][j]:
                    stack = [(i, j, 0, [])]
                    visited[i][j] = True
                    while stack:
                        x, y, steps, path = stack.pop()
                        if (x, y) in path:
                            has_infinite = True
                            break
                        new_path = path + [(x, y)]
                        current_char = grid[x][y]
                        next_c = next_char(current_char)
                        for dx, dy in directions:
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < n and 0 <= ny < m:
                                if grid[nx][ny] == next_c and not visited[nx][ny]:
                                    visited[nx][ny] = True
                                    stack.append((nx, ny, steps + 1, new_path))
                        dp[x][y] = max(dp[x][y], steps // 4 + 1)
                    visited[i][j] = False
                    if has_infinite:
                        break
            if has_infinite:
                break

        if has_infinite:
            return "Poor Inna!"
        else:
            max_dima = 0
            for i in range(n):
                for j in range(m):
                    if dp[i][j] > max_dima:
                        max_dima = dp[i][j]
            if max_dima < 4:
                return "Poor Dima!"
            else:
                return str(max_dima // 4)
