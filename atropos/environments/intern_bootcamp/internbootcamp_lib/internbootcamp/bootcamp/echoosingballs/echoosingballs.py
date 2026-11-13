"""# 

### 谜题描述
There are n balls. They are arranged in a row. Each ball has a color (for convenience an integer) and an integer value. The color of the i-th ball is ci and the value of the i-th ball is vi.

Squirrel Liss chooses some balls and makes a new sequence without changing the relative order of the balls. She wants to maximize the value of this sequence.

The value of the sequence is defined as the sum of following values for each ball (where a and b are given constants):

  * If the ball is not in the beginning of the sequence and the color of the ball is same as previous ball's color, add (the value of the ball)  ×  a. 
  * Otherwise, add (the value of the ball)  ×  b. 



You are given q queries. Each query contains two integers ai and bi. For each query find the maximal value of the sequence she can make when a = ai and b = bi.

Note that the new sequence can be empty, and the value of an empty sequence is defined as zero.

Input

The first line contains two integers n and q (1 ≤ n ≤ 105; 1 ≤ q ≤ 500). The second line contains n integers: v1, v2, ..., vn (|vi| ≤ 105). The third line contains n integers: c1, c2, ..., cn (1 ≤ ci ≤ n).

The following q lines contain the values of the constants a and b for queries. The i-th of these lines contains two integers ai and bi (|ai|, |bi| ≤ 105).

In each line integers are separated by single spaces.

Output

For each query, output a line containing an integer — the answer to the query. The i-th line contains the answer to the i-th query in the input order.

Please, do not write the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

6 3
1 -2 3 4 0 -1
1 2 1 2 1 1
5 1
-2 1
1 0


Output

20
9
4


Input

4 1
-3 6 -1 2
1 2 3 1
1 -1


Output

5

Note

In the first example, to achieve the maximal value:

  * In the first query, you should select 1st, 3rd, and 4th ball. 
  * In the second query, you should select 3rd, 4th, 5th and 6th ball. 
  * In the third query, you should select 2nd and 4th ball. 



Note that there may be other ways to achieve the maximal value.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n, v[100000], c[100000];
int q, a, b;
long long f[100000 + 1];
bool m[100000 + 1];
int x, y;
long long query(long long a, long long b) {
  memset(f, 0, sizeof(f));
  memset(m, 0, sizeof(m));
  f[0] = x = y = 0;
  long long res = 0;
  for (int i = 0; i < n; ++i) {
    long long other = c[i] == x ? f[y] : f[x];
    if (m[c[i]]) {
      long long old = f[c[i]];
      f[c[i]] = max(old, max(old + v[i] * a, other + v[i] * b));
    } else {
      f[c[i]] = other + v[i] * b;
      m[c[i]] = 1;
    }
    res = max(res, f[c[i]]);
    if (c[i] == x || c[i] == y) {
      if (f[x] < f[y]) {
        int tmp = x;
        x = y;
        y = tmp;
      }
    } else {
      if (f[c[i]] > f[x]) {
        y = x;
        x = c[i];
      } else if (f[c[i]] > f[y]) {
        y = c[i];
      }
    }
  }
  return res;
}
int main(int argc, char *argv[]) {
  cin >> n >> q;
  for (int i = 0; i < n; ++i) {
    cin >> v[i];
  }
  for (int i = 0; i < n; ++i) {
    cin >> c[i];
  }
  for (int i = 0; i < q; ++i) {
    cin >> a >> b;
    cout << query(a, b) << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Echoosingballsbootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=6, v_min=-5, v_max=5, c_max=3, q_min=1, q_max=3):
        self.n_min = n_min
        self.n_max = n_max
        self.v_min = v_min
        self.v_max = v_max
        self.c_max = c_max
        self.q_min = q_min
        self.q_max = q_max

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        v = [random.randint(self.v_min, self.v_max) for _ in range(n)]
        c = [random.randint(1, self.c_max) for _ in range(n)]
        q = random.randint(self.q_min, self.q_max)
        queries = [(random.randint(-5, 5), random.randint(-5, 5)) for _ in range(q)]
        expected_outputs = [self._compute_query(v, c, a, b) for a, b in queries]
        return {
            'n': n,
            'q': q,
            'v': v,
            'c': c,
            'queries': queries,
            'expected_outputs': expected_outputs
        }

    @staticmethod
    def _compute_query(v_list, c_list, a, b):
        f = {}
        x = None  # Largest color
        y = None  # Second largest color
        res = 0
        
        for vi, ci in zip(v_list, c_list):
            other = 0
            if x is not None:
                if ci == x:
                    other = f.get(y, 0) if y is not None else 0
                else:
                    other = f.get(x, 0)

            current = f.get(ci, -float('inf'))
            new_val = other + vi * b
            if current != -float('inf'):
                new_val = max(new_val, current + vi * a)
            
            f[ci] = max(current, new_val) if current != -float('inf') else new_val
            res = max(res, f[ci])

            # Update color rankings
            colors = sorted(f.items(), key=lambda x: -x[1])
            x = colors[0][0] if colors else None
            y = colors[1][0] if len(colors) > 1 else None

        return max(res, 0)

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        q = question_case['q']
        v_str = ' '.join(map(str, question_case['v']))
        c_str = ' '.join(map(str, question_case['c']))
        queries = '\n'.join([f"{a} {b}" for a, b in question_case['queries']])
        return f"""你是一个算法竞赛选手，需要解决以下问题：

给定{n}个球按序排列，每个球有颜色c和值v。现在有{q}个查询，每个查询给出系数a和b。要求对每个查询找出最大子序列值。规则如下：
1. 子序列保持原顺序
2. 第一个球的贡献为v_i × b
3. 后续球如果颜色与前一个相同，贡献为v_i × a，否则为v_i × b

输入数据：
n = {n}, q = {q}
v数组: {v_str}
c数组: {c_str}
查询列表:
{queries}

请计算每个查询的最大价值，并按顺序将结果用换行分隔放在[answer]标签内，例如：
[answer]
结果1
结果2
[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answers = []
        for line in matches[-1].strip().split('\n'):
            line = line.strip()
            if line:
                try:
                    answers.append(int(line))
                except:
                    continue
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_outputs']
        return isinstance(solution, list) and len(solution) == len(expected) and all(s == e for s, e in zip(solution, expected))
