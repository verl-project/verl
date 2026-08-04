"""# 

### 谜题描述
After years of hard work scientists invented an absolutely new e-reader display. The new display has a larger resolution, consumes less energy and its production is cheaper. And besides, one can bend it. The only inconvenience is highly unusual management. For that very reason the developers decided to leave the e-readers' software to programmers.

The display is represented by n × n square of pixels, each of which can be either black or white. The display rows are numbered with integers from 1 to n upside down, the columns are numbered with integers from 1 to n from the left to the right. The display can perform commands like \"x, y\". When a traditional display fulfills such command, it simply inverts a color of (x, y), where x is the row number and y is the column number. But in our new display every pixel that belongs to at least one of the segments (x, x) - (x, y) and (y, y) - (x, y) (both ends of both segments are included) inverts a color.

For example, if initially a display 5 × 5 in size is absolutely white, then the sequence of commands (1, 4), (3, 5), (5, 1), (3, 3) leads to the following changes:

<image>

You are an e-reader software programmer and you should calculate minimal number of commands needed to display the picture. You can regard all display pixels as initially white.

Input

The first line contains number n (1 ≤ n ≤ 2000).

Next n lines contain n characters each: the description of the picture that needs to be shown. \"0\" represents the white color and \"1\" represents the black color. 

Output

Print one integer z — the least number of commands needed to display the picture.

Examples

Input

5
01110
10010
10001
10011
11110


Output

4

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int nmax = 2009;
int a[2009][2009];
int ui[nmax], uj[nmax];
int n;
void inc(int &a) {
  ++a;
  a = a & 1;
}
int main() {
  int ans = 0;
  scanf(\"%d\n\", &n);
  char s[nmax];
  for (int i = 1; i <= n; i++) {
    scanf(\"%s\", s);
    for (int j = 1; j <= n; j++) a[i][j] = s[j - 1] - '0';
  }
  for (int k = n; k >= 1; k--) {
    int p = 0;
    for (int i = 1; i < k; i++) {
      if ((p & 1) == 1) inc(a[i][k]);
      if ((ui[i] & 1) == 1) inc(a[i][k]);
      if (a[i][k] != 0) {
        inc(p);
        inc(ui[i]);
        inc(a[i][k]);
        ans++;
      }
    }
    if ((p & 1) == 1) inc(a[k][k]);
    if ((ui[k] & 1) == 1) inc(a[k][k]);
    p = 0;
    for (int j = 1; j < k; j++) {
      if ((p & 1) == 1) inc(a[k][j]);
      if ((uj[j] & 1) == 1) inc(a[k][j]);
      if (a[k][j] != 0) {
        inc(p);
        inc(uj[j]);
        inc(a[k][j]);
        ans++;
      }
    }
    if ((p & 1) == 1) inc(a[k][k]);
    if ((uj[k] & 1) == 1) inc(a[k][k]);
    if (a[k][k] != 0) {
      ans++;
      inc(a[k][k]);
    }
  }
  cout << ans << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def apply_command(grid, x, y):
    """应用命令(x, y)到初始网格，翻转对应像素"""
    n = len(grid)
    min_col = min(x, y)
    max_col = max(x, y)
    # 翻转行x[min_col..max_col]
    for c in range(min_col, max_col + 1):
        grid[x-1][c-1] ^= 1
    
    min_row = min(x, y)
    max_row = max(x, y)
    # 翻转列y[min_row..max_row]
    for r in range(min_row, max_row + 1):
        grid[r-1][y-1] ^= 1

def calculate_min_commands(n, grid):
    """参考C++代码的Python实现，计算最小命令次数"""
    a = [[0]*(n+2) for _ in range(n+2)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            a[i][j] = int(grid[i-1][j-1])
    
    ui = [0]*(n+2)
    uj = [0]*(n+2)
    ans = 0

    for k in range(n, 0, -1):
        p = 0
        # 处理行部分
        for i in range(1, k):
            if p % 2 == 1:
                a[i][k] ^= 1
            if ui[i] % 2 == 1:
                a[i][k] ^= 1
            if a[i][k] != 0:
                p ^= 1
                ui[i] ^= 1
                a[i][k] = 0
                ans += 1
        
        if p % 2 == 1:
            a[k][k] ^= 1
        if ui[k] % 2 == 1:
            a[k][k] ^= 1
        
        p = 0
        # 处理列部分
        for j in range(1, k):
            if p % 2 == 1:
                a[k][j] ^= 1
            if uj[j] % 2 == 1:
                a[k][j] ^= 1
            if a[k][j] != 0:
                p ^= 1
                uj[j] ^= 1
                a[k][j] = 0
                ans += 1
        
        if p % 2 == 1:
            a[k][k] ^= 1
        if uj[k] % 2 == 1:
            a[k][k] ^= 1
        
        if a[k][k] != 0:
            ans += 1
            a[k][k] = 0
    
    return ans

class Eereaderdisplaybootcamp(Basebootcamp):
    def __init__(self, n=5):
        """初始化训练场参数，默认创建5x5的问题"""
        self.n = n
    
    def case_generator(self):
        """生成有效问题实例，保证有解"""
        n = self.n
        # 初始化全白网格（使用整数矩阵方便操作）
        grid = [[0]*n for _ in range(n)]
        # 生成随机命令序列（1-2n次命令）
        for _ in range(random.randint(1, 2*n)):
            x = random.randint(1, n)
            y = random.randint(1, n)
            apply_command(grid, x, y)
        
        # 转换为字符串格式
        grid_str = [''.join(map(str, row)) for row in grid]
        # 计算标准答案
        answer = calculate_min_commands(n, grid_str)
        
        return {
            'n': n,
            'grid': grid_str,
            'answer': answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """生成符合格式的问题描述"""
        n = question_case['n']
        grid = '\n'.join(question_case['grid'])
        return f"""你是电子墨水屏软件开发人员，需要计算显示指定图案所需的最少命令。显示器由{n}x{n}像素构成，初始全白。每个命令(x,y)会翻转两条线段上的所有像素：
1. 行x从列min(x,y)到max(x,y)
2. 列y从行min(x,y)到max(x,y)

目标图案：
{n}
{grid}

请计算所需最少命令次数，并将答案放入[answer]标签内，如：[answer]5[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        """从回复中提取最后一个答案"""
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """验证答案是否正确"""
        return solution == identity['answer']
