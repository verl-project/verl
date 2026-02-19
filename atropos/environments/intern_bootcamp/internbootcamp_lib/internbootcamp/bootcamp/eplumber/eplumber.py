"""# 

### 谜题描述
Little John aspires to become a plumber! Today he has drawn a grid consisting of n rows and m columns, consisting of n × m square cells.

In each cell he will draw a pipe segment. He can only draw four types of segments numbered from 1 to 4, illustrated as follows:

<image>

Each pipe segment has two ends, illustrated by the arrows in the picture above. For example, segment 1 has ends at top and left side of it.

Little John considers the piping system to be leaking if there is at least one pipe segment inside the grid whose end is not connected to another pipe's end or to the border of the grid. The image below shows an example of leaking and non-leaking systems of size 1 × 2.

<image>

Now, you will be given the grid that has been partially filled by Little John. Each cell will either contain one of the four segments above, or be empty. Find the number of possible different non-leaking final systems after Little John finishes filling all of the empty cells with pipe segments. Print this number modulo 1000003 (106 + 3).

Note that rotations or flipping of the grid are not allowed and so two configurations that are identical only when one of them has been rotated or flipped either horizontally or vertically are considered two different configurations.

Input

The first line will contain two single-space separated integers n and m (1 ≤ n, m, n·m ≤ 5·105) — the number of rows and columns respectively. Then n lines follow, each contains exactly m characters — the description of the grid. Each character describes a cell and is either one of these: 

  * \"1\" - \"4\" — a pipe segment of one of four types as described above 
  * \".\" — an empty cell 

Output

Print a single integer denoting the number of possible final non-leaking pipe systems modulo 1000003 (106 + 3). If there are no such configurations, print 0.

Examples

Input

2 2
13
..


Output

2


Input

3 1
1
4
.


Output

0


Input

2 2
3.
.1


Output

1

Note

For the first example, the initial configuration of the grid is as follows. 

<image>

The only two possible final non-leaking pipe configurations are as follows:

<image> <image>

For the second example, the initial grid is already leaking, so there will be no final grid that is non-leaking.

For the final example, there's only one possible non-leaking final grid as follows.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int mod = int(1e6 + 3);
vector<vector<int> > v;
char second[600000];
int n, m;
int main() {
  scanf(\"%d%d\n\", &n, &m);
  v.resize(n);
  for (int i = 0; i < n; ++i) {
    gets(second);
    v[i].resize(m);
    for (int j = 0; j < m; ++j) {
      v[i][j] = (second[j] == '.') ? -1 : second[j] - '0';
    }
  }
  int res = 1;
  for (int i = 0; i < n; ++i) {
    int dir = -1;
    for (int j = 0; j < m; ++j)
      if (v[i][j] != -1) {
        dir = (v[i][j] > 2) ? 1 : 0;
        if (j & 1) dir = 1 - dir;
        break;
      }
    if (dir != -1) {
      for (int j = 0; j < m; ++j)
        if (v[i][j] != -1) {
          int d2 = (v[i][j] > 2) ? 1 : 0;
          if (j & 1) d2 = 1 - d2;
          if (dir != d2) res = 0;
        }
    } else if (dir == -1)
      res = (res * 2) % mod;
  }
  for (int j = 0; j < m; ++j) {
    int dir = -1;
    for (int i = 0; i < n; ++i)
      if (v[i][j] != -1) {
        dir = (v[i][j] > 1 && v[i][j] < 4) ? 1 : 0;
        if (i & 1) dir = 1 - dir;
        break;
      }
    if (dir != -1) {
      for (int i = 0; i < n; ++i)
        if (v[i][j] != -1) {
          int d2 = (v[i][j] > 1 && v[i][j] < 4) ? 1 : 0;
          if (i & 1) d2 = 1 - d2;
          if (dir != d2) res = 0;
        }
    } else if (dir == -1)
      res = (res * 2) % mod;
  }
  printf(\"%d\n\", res);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Eplumberbootcamp(Basebootcamp):
    def __init__(self, max_size=5, **kwargs):
        super().__init__(**kwargs)
        self.max_size = max_size  # 控制生成网格的最大行列数
        self.mod = 1000003

    def case_generator(self):
        mod = self.mod
        # 生成有效的案例，随机决定是否引入冲突
        n = random.randint(1, self.max_size)
        m = random.randint(1, self.max_size)
        if n * m > 500000:  # 保证n*m <=5e5
            n = m = 1

        # 初始化网格全为未填充
        grid = [['.' for _ in range(m)] for _ in range(n)]
        
        # 随机选择是否生成冲突案例（30%概率生成冲突）
        has_conflict = random.random() < 0.3

        # 随机确定每行和每列的约束方向
        row_dir = {}  # 被约束行的方向
        col_dir = {}  # 被约束列的方向
        
        constrained_rows = set()
        constrained_cols = set()

        # 随机选择部分行作为被约束的
        for i in range(n):
            if random.random() < 0.5 and n > 1:
                constrained_rows.add(i)
                row_dir[i] = random.choice([0, 1])
        
        # 随机选择部分列作为被约束的
        for j in range(m):
            if random.random() < 0.5 and m > 1:
                constrained_cols.add(j)
                col_dir[j] = random.choice([0, 1])

        # 填充部分单元格以满足约束条件
        for i in constrained_rows:
            # 选择该行中一个列来填充
            j = random.choice([j for j in range(m)])
            possible = self.get_possible_types(i, j, row_dir[i], col_dir.get(j, None))
            if not possible and j in col_dir:
                # 如果列也被约束，可能无法找到可能的类型，需要调整
                possible = self.get_possible_types(i, j, row_dir[i], col_dir[j])
                if not possible:
                    # 如果无法找到类型，可能需调整dir
                    row_dir[i] = 1 - row_dir[i]
                    possible = self.get_possible_types(i, j, row_dir[i], col_dir.get(j, None))
            if possible:
                grid[i][j] = str(random.choice(possible))
            else:
                # 无法满足约束，重新生成
                return self.case_generator()

        for j in constrained_cols:
            i = random.choice([i for i in range(n)])
            possible = self.get_possible_types(i, j, row_dir.get(i, None), col_dir[j])
            if not possible and i in row_dir:
                possible = self.get_possible_types(i, j, row_dir[i], col_dir[j])
                if not possible:
                    col_dir[j] = 1 - col_dir[j]
                    possible = self.get_possible_types(i, j, row_dir.get(i, None), col_dir[j])
            if possible:
                grid[i][j] = str(random.choice(possible))
            else:
                return self.case_generator()

        # 引入冲突（如果有冲突标志）
        if has_conflict and (constrained_rows or constrained_cols):
            # 随机选择一个已填充的单元格修改为冲突类型
            filled = [(i,j) for i in range(n) for j in range(m) if grid[i][j] != '.']
            if filled:
                i,j = random.choice(filled)
                current_type = int(grid[i][j])
                possible = self.get_possible_types(i, j, row_dir.get(i, None), col_dir.get(j, None))
                # 选择不在possible中的类型（确保冲突）
                conflict_types = [str(t) for t in [1,2,3,4] if t != current_type and (possible is None or t not in possible)]
                if conflict_types:
                    grid[i][j] = random.choice(conflict_types)
                else:
                    # 无法生成冲突，处理这种情况
                    has_conflict = False

        # 计算未被约束的行和列数
        unconstrained_rows = n - len(constrained_rows)
        unconstrained_cols = m - len(constrained_cols)
        expected = pow(2, unconstrained_rows + unconstrained_cols, mod)
        
        # 检查是否存在冲突
        if has_conflict:
            expected = 0

        # 将网格转换为字符串列表
        grid_str = [''.join(row) for row in grid]
        return {
            'n': n,
            'm': m,
            'grid': grid_str,
            'expected': expected
        }

    @staticmethod
    def get_possible_types(i, j, row_dir, col_dir):
        # 根据行和列约束生成可能的类型，None表示无约束
        possible = set([1,2,3,4])
        # 行约束
        if row_dir is not None:
            if row_dir == 0:
                if j % 2 == 0:
                    row_possible = {1,2}
                else:
                    row_possible = {3,4}
            else: # row_dir ==1
                if j % 2 ==0:
                    row_possible = {3,4}
                else:
                    row_possible = {1,2}
            possible &= row_possible
        # 列约束
        if col_dir is not None:
            if col_dir ==0:
                if i %2 ==0:
                    col_possible = {1,4}
                else:
                    col_possible = {2,3}
            else: # col_dir ==1
                if i%2 ==0:
                    col_possible = {2,3}
                else:
                    col_possible = {1,4}
            possible &= col_possible
        return list(possible) if possible else None

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        grid = '\n'.join(question_case['grid'])
        prompt = f"""Little John 绘制了一个 {n} 行 {m} 列的网格，每个单元格可能为空（.）或包含1-4号水管。系统漏水当且仅当某管道的末端未连接其他管道或边界。请计算填充所有空单元格后，可能的不漏水方案数目模1000003。

输入:
{n} {m}
{grid}

规则：
- 水管类型1-4的末端方向如题图所示。
- 相邻单元格的末端必须匹配，例如，右边的单元格的左端必须连接当前单元格的右端。
- 网格边缘的管道末端需指向外部以避免漏水。

输出一个整数，模1000003后的结果，置于[answer]标签内。例如：[answer]0[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
