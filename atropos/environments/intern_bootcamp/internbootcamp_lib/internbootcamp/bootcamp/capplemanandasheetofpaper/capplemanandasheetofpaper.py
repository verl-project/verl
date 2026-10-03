"""# 

### 谜题描述
Appleman has a very big sheet of paper. This sheet has a form of rectangle with dimensions 1 × n. Your task is help Appleman with folding of such a sheet. Actually, you need to perform q queries. Each query will have one of the following types:

  1. Fold the sheet of paper at position pi. After this query the leftmost part of the paper with dimensions 1 × pi must be above the rightmost part of the paper with dimensions 1 × ([current width of sheet] - pi). 
  2. Count what is the total width of the paper pieces, if we will make two described later cuts and consider only the pieces between the cuts. We will make one cut at distance li from the left border of the current sheet of paper and the other at distance ri from the left border of the current sheet of paper. 



Please look at the explanation of the first test example for better understanding of the problem.

Input

The first line contains two integers: n and q (1 ≤ n ≤ 105; 1 ≤ q ≤ 105) — the width of the paper and the number of queries.

Each of the following q lines contains one of the described queries in the following format:

  * \"1 pi\" (1 ≤ pi < [current width of sheet]) — the first type query. 
  * \"2 li ri\" (0 ≤ li < ri ≤ [current width of sheet]) — the second type query. 

Output

For each query of the second type, output the answer.

Examples

Input

7 4
1 3
1 2
2 0 1
2 1 2


Output

4
3


Input

10 9
2 2 9
1 1
2 0 1
1 8
2 0 8
1 2
2 1 3
1 4
2 2 4


Output

7
2
10
4
5

Note

The pictures below show the shapes of the paper during the queries of the first example:

<image>

After the first fold operation the sheet has width equal to 4, after the second one the width of the sheet equals to 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e5 + 10;
int C[maxn];
int n;
int inc;
int lowbit(int x) { return x & -x; }
void add(int x, int v) {
  while (x <= n + inc) {
    C[x] += v;
    x = x + lowbit(x);
  }
}
int sum(int x) {
  int res = 0;
  while (x > 0) {
    res += C[x];
    x = x - lowbit(x);
  }
  return res;
}
int main() {
  int Q;
  scanf(\"%d%d\", &n, &Q);
  inc = 0;
  int flip = 0;
  for (int i = 1; i <= n; i++) add(i, 1);
  for (int _Q = 1; _Q <= Q; ++_Q) {
    int q;
    scanf(\"%d\", &q);
    if (q == 1) {
      int x;
      scanf(\"%d\", &x);
      if (flip) x = n - x;
      if (x > n - x) {
        flip = 1;
        for (int i = x, j = x + 1; j <= n; j++, i--) {
          int c = sum(inc + j) - sum(inc + j - 1);
          add(inc + i, c);
        }
        n = x;
      } else {
        if (flip) flip = 0;
        for (int i = x, j = x + 1; i >= 1; i--, j++) {
          int c = sum(inc + i) - sum(inc + i - 1);
          add(inc + j, c);
        }
        inc += x;
        n = n - x;
      }
    } else {
      int l, r;
      scanf(\"%d%d\", &l, &r);
      l++;
      if (flip) {
        int t = l;
        l = n - r + 1;
        r = n - t + 1;
      }
      printf(\"%d\n\", sum(r + inc) - sum(l + inc - 1));
    }
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

class FenwickTree:
    def __init__(self, max_size):
        self.N = max_size
        self.tree = [0] * (self.N + 2)

    def update(self, idx, delta):
        while idx <= self.N:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def generate_operations(n, q):
    current_width = n
    flip = False
    operations = []
    for _ in range(q):
        if current_width <= 1:
            op_type = 2
        else:
            op_type = random.choice([1, 2])
        if op_type == 1:
            pi = random.randint(1, current_width - 1)
            x = current_width - pi if flip else pi
            if x > current_width - x:
                new_width = x
                new_flip = True
            else:
                new_width = current_width - x
                new_flip = False
            operations.append((1, pi))
            current_width = new_width
            flip = new_flip
        else:
            if current_width == 0:
                li, ri = 0, 0
            else:
                li = random.randint(0, current_width - 1)
                ri = random.randint(li + 1, current_width)
            operations.append((2, li, ri))
    return operations

def solve_case(n, q, operations):
    max_size = 200000
    ft = FenwickTree(max_size)
    inc = 0
    flip = False
    current_n = n
    for j in range(1, n + 1):
        ft.update(j, 1)
    outputs = []
    for query in operations:
        if query[0] == 1:
            p = query[1]
            x = current_n - p if flip else p
            if x > current_n - x:
                flip = True
                new_n = x
                j_start = x + 1
                j_end = current_n
                i, j = x, j_start
                while j <= j_end and i >= 1:
                    val = ft.query(inc + j) - ft.query(inc + j - 1)
                    ft.update(inc + i, val)
                    i -= 1
                    j += 1
                current_n = new_n
            else:
                flip = False
                i, j = x, x + 1
                while i >= 1 and j <= current_n:
                    val = ft.query(inc + i) - ft.query(inc + i - 1)
                    ft.update(inc + j, val)
                    i -= 1
                    j += 1
                inc += x
                current_n -= x
        else:
            li, ri = query[1], query[2]
            l = li + 1
            r = ri
            if flip:
                new_l = current_n - r + 1
                new_r = current_n - (l - 1)
                l, r = new_l, new_r
            left = inc + l
            right = inc + r
            res = ft.query(right) - ft.query(left - 1)
            outputs.append(res)
    return outputs

class Capplemanandasheetofpaperbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_q=10):
        self.max_n = max_n
        self.max_q = max_q

    def case_generator(self):
        n = random.randint(1, self.max_n)
        q = random.randint(1, self.max_q)
        operations = generate_operations(n, q)
        expected_outputs = solve_case(n, q, operations)
        while len(expected_outputs) != sum(1 for op in operations if op[0] == 2):
            n = random.randint(1, self.max_n)
            q = random.randint(1, self.max_q)
            operations = generate_operations(n, q)
            expected_outputs = solve_case(n, q, operations)
        return {
            'n': n,
            'q': q,
            'operations': operations,
            'expected_outputs': expected_outputs
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['q']}"]
        for op in question_case['operations']:
            if op[0] == 1:
                input_lines.append(f"1 {op[1]}")
            else:
                input_lines.append(f"2 {op[1]} {op[2]}")
        input_str = '\n'.join(input_lines)
        prompt = f"""
Appleman有一个1×{question_case['n']}的纸条，需处理{question_case['q']}个查询：
1. 折叠：在位置pi折叠，左边部分覆盖右边，纸张宽度变为折叠后的当前值。
2. 查询：在距离左边界li和ri的位置切割，求之间的总层数。

输入：
{input_str}

请输出每个类型2查询的结果，按顺序每行一个，置于[answer]和[/answer]之间。例如：
[answer]
结果1
结果2
[/answer]
"""
        return prompt.strip()

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        try:
            return list(map(int, last_answer.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_outputs']
