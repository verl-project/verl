"""# 

### 谜题描述
Ilya is sitting in a waiting area of Metropolis airport and is bored of looking at time table that shows again and again that his plane is delayed. So he took out a sheet of paper and decided to solve some problems.

First Ilya has drawn a grid of size n × n and marked n squares on it, such that no two marked squares share the same row or the same column. He calls a rectangle on a grid with sides parallel to grid sides beautiful if exactly two of its corner squares are marked. There are exactly n·(n - 1) / 2 beautiful rectangles.

Ilya has chosen q query rectangles on a grid with sides parallel to grid sides (not necessarily beautiful ones), and for each of those rectangles he wants to find its beauty degree. Beauty degree of a rectangle is the number of beautiful rectangles that share at least one square with the given one.

Now Ilya thinks that he might not have enough time to solve the problem till the departure of his flight. You are given the description of marked cells and the query rectangles, help Ilya find the beauty degree of each of the query rectangles.

Input

The first line of input contains two integers n and q (2 ≤ n ≤ 200 000, 1 ≤ q ≤ 200 000) — the size of the grid and the number of query rectangles.

The second line contains n integers p1, p2, ..., pn, separated by spaces (1 ≤ pi ≤ n, all pi are different), they specify grid squares marked by Ilya: in column i he has marked a square at row pi, rows are numbered from 1 to n, bottom to top, columns are numbered from 1 to n, left to right.

The following q lines describe query rectangles. Each rectangle is described by four integers: l, d, r, u (1 ≤ l ≤ r ≤ n, 1 ≤ d ≤ u ≤ n), here l and r are the leftmost and the rightmost columns of the rectangle, d and u the bottommost and the topmost rows of the rectangle.

Output

For each query rectangle output its beauty degree on a separate line.

Examples

Input

2 3
1 2
1 1 1 1
1 1 1 2
1 1 2 2


Output

1
1
1


Input

4 2
1 3 2 4
4 1 4 4
1 1 2 3


Output

3
5

Note

The first sample test has one beautiful rectangle that occupies the whole grid, therefore the answer to any query is 1.

In the second sample test the first query rectangle intersects 3 beautiful rectangles, as shown on the picture below:

<image> <image> <image>

There are 5 beautiful rectangles that intersect the second query rectangle, as shown on the following picture:

<image> <image> <image> <image> <image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long N = 200005, logN = 21;
struct data {
  long long ls, rs, val;
} tree[N * logN];
long long cur, rt[N], n, q, sortb[N], b[N];
inline void init() { cur = 0; }
inline void push_up(long long p) {
  tree[p].val = tree[tree[p].ls].val + tree[tree[p].rs].val;
}
inline long long build(long long l, long long r) {
  long long k = cur++;
  if (l == r) {
    tree[k].val = 0;
    return k;
  }
  long long mid = (l + r) >> 1;
  tree[k].ls = build(l, mid);
  tree[k].rs = build(mid + 1, r);
  push_up(k);
  return k;
}
inline long long insert(long long nod, long long l, long long r, long long pos,
                        long long add) {
  long long k = cur++;
  tree[k] = tree[nod];
  if (l == r) {
    tree[k].val += add;
    return k;
  }
  long long mid = (l + r) >> 1;
  if (pos <= mid)
    tree[k].ls = insert(tree[nod].ls, l, mid, pos, add);
  else
    tree[k].rs = insert(tree[nod].rs, mid + 1, r, pos, add);
  push_up(k);
  return k;
}
inline long long ask(long long l, long long r, long long i, long long j,
                     long long nod) {
  if (i > j) return 0;
  if (l == i && r == j) return tree[nod].val;
  long long mid = l + r >> 1;
  if (j <= mid)
    return ask(l, mid, i, j, tree[nod].ls);
  else if (i > mid)
    return ask(mid + 1, r, i, j, tree[nod].rs);
  else
    return ask(l, mid, i, mid, tree[nod].ls) +
           ask(mid + 1, r, mid + 1, j, tree[nod].rs);
}
inline long long read() {
  long long x = 0;
  char ch = getchar();
  bool positive = 1;
  for (; !isdigit(ch); ch = getchar())
    if (ch == '-') positive = 0;
  for (; isdigit(ch); ch = getchar()) x = x * 10 + ch - '0';
  return positive ? x : -x;
}
inline void write(long long a) {
  if (a >= 10) write(a / 10);
  putchar('0' + a % 10);
}
inline void writeln(long long a) {
  if (a < 0) {
    putchar('-');
    a = -a;
  }
  write(a);
  puts(\"\");
}
long long an[N][9];
int main() {
  n = read();
  long long m = read();
  init();
  rt[0] = build(1, n);
  for (long long i = 1; i <= n; i++) {
    rt[i] = insert(rt[i - 1], 1, n, read(), 1);
  }
  for (long long i = 1; i <= m; i++) {
    long long l = read(), d = read(), r = read(), h = read();
    an[i][0] = ask(1, n, 1, d - 1, rt[l - 1]);
    an[i][1] = ask(1, n, 1, h, rt[l - 1]);
    an[i][2] = ask(1, n, 1, n, rt[l - 1]);
    an[i][3] = ask(1, n, 1, d - 1, rt[r]) - an[i][0];
    an[i][4] = ask(1, n, 1, h, rt[r]) - an[i][1];
    an[i][5] = ask(1, n, 1, n, rt[r]) - an[i][2];
    an[i][6] = ask(1, n, 1, d - 1, rt[n]) - an[i][0] - an[i][3];
    an[i][7] = ask(1, n, 1, h, rt[n]) - an[i][1] - an[i][4];
    an[i][8] = ask(1, n, 1, n, rt[n]) - an[i][2] - an[i][5];
    an[i][1] -= an[i][0];
    an[i][2] -= an[i][0] + an[i][1];
    an[i][4] -= an[i][3];
    an[i][5] -= an[i][3] + an[i][4];
    an[i][7] -= an[i][6];
    an[i][8] -= an[i][6] + an[i][7];
  }
  for (long long i = 1; i <= m; i++) {
    writeln(an[i][0] * (an[i][4] + an[i][5] + an[i][7] + an[i][8]) +
            (long long)an[i][1] * (n - an[i][1] - an[i][2] - an[i][0]) +
            (long long)an[i][2] * (an[i][3] + an[i][4] + an[i][6] + an[i][7]) +
            (long long)an[i][3] * (an[i][4] + an[i][5] + an[i][7] + an[i][8]) +
            (long long)an[i][4] * (an[i][5] + an[i][6] + an[i][7] + an[i][8]) +
            (long long)an[i][5] * (an[i][6] + an[i][7]) +
            (long long)an[i][4] * (an[i][4] - 1) / 2);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cboredombootcamp(Basebootcamp):
    def __init__(self, n=4, q=2, **kwargs):
        """
        初始化谜题训练场，配置网格大小和查询数量。
        参数:
            n: 网格大小，默认4
            q: 查询数量，默认2
        """
        super().__init__(**kwargs)
        self.n = n
        self.q = q

    def case_generator(self):
        """生成符合谜题要求的测试实例，包含输入数据和正确答案"""
        # 生成随机排列的标记点位置
        p = list(range(1, self.n + 1))
        random.shuffle(p)
        
        # 随机生成查询矩形
        queries = []
        for _ in range(self.q):
            l = random.randint(1, self.n)
            r = random.randint(l, self.n)
            d = random.randint(1, self.n)
            u = random.randint(d, self.n)
            queries.append((l, d, r, u))
        
        # 计算每个查询的正确答案
        answers = self._calculate_answers(p, queries)
        
        return {
            'n': self.n,
            'q': self.q,
            'p': p,
            'queries': queries,
            'answers': answers
        }

    def _calculate_answers(self, p, queries):
        """暴力计算每个查询的正确答案（适用于小规模数据）"""
        answers = []
        # 预先生成所有美丽矩形的特征
        beautiful_rects = []
        for i in range(1, self.n + 1):
            for j in range(i + 1, self.n + 1):
                row_i = p[i - 1]
                row_j = p[j - 1]
                min_row = min(row_i, row_j)
                max_row = max(row_i, row_j)
                beautiful_rects.append((i, j, min_row, max_row))
        
        # 处理每个查询
        for (l, d, r, u) in queries:
            count = 0
            for (col1, col2, min_r, max_r) in beautiful_rects:
                # 列重叠检查
                if not (col1 <= r and col2 >= l):
                    continue
                # 行重叠检查
                if not (min_r <= u and max_r >= d):
                    continue
                count += 1
            answers.append(count)
        return answers

    @staticmethod
    def prompt_func(question_case) -> str:
        """将问题实例转换为自然语言描述"""
        p = question_case['p']
        queries = question_case['queries']
        n = question_case['n']
        q = question_case['q']
        
        problem_text = f"""Ilya需要解决一个关于{n}x{n}网格的问题。网格中每一列有一个标记，且每行每列恰好一个标记。每个查询需要计算与指定矩形相交的"美丽矩形"数量。

输入格式：
- 第一行：{n} {q}
- 第二行：{' '.join(map(str, p))}
- 随后{q}行，每行四个整数描述矩形：l d r u（分别表示左、底、右、顶）

请依次输出每个查询对应的美丽矩形数量，每个答案单独一行，包裹在[answer]标签中。

示例输入：
{n} {q}
{' '.join(map(str, p))}
"""
        for qry in queries:
            problem_text += f"{' '.join(map(str, qry))}\n"
        
        problem_text += "\n请将最终答案按顺序放在[answer]和[/answer]之间，例如：\n[answer]\n3\n5\n[/answer]"
        return problem_text

    @staticmethod
    def extract_output(output):
        """从模型输出中提取答案列表"""
        # 使用非贪婪匹配找到所有答案块
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL | re.IGNORECASE)
        if not answer_blocks:
            return None
        
        # 取最后一个答案块并解析
        last_block = answer_blocks[-1].strip()
        answers = []
        for line in last_block.split('\n'):
            line = line.strip()
            if line:
                try:
                    answers.append(int(line))
                except ValueError:
                    continue
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """验证答案是否正确"""
        if solution is None:
            return False
        return solution == identity['answers']
