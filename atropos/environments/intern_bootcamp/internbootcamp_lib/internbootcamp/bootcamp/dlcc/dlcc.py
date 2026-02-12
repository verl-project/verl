"""# 

### 谜题描述
An infinitely long Line Chillland Collider (LCC) was built in Chillland. There are n pipes with coordinates x_i that are connected to LCC. When the experiment starts at time 0, i-th proton flies from the i-th pipe with speed v_i. It flies to the right with probability p_i and flies to the left with probability (1 - p_i). The duration of the experiment is determined as the time of the first collision of any two protons. In case there is no collision, the duration of the experiment is considered to be zero.

Find the expected value of the duration of the experiment.

<image>Illustration for the first example

Input

The first line of input contains one integer n — the number of pipes (1 ≤ n ≤ 10^5). Each of the following n lines contains three integers x_i, v_i, p_i — the coordinate of the i-th pipe, the speed of the i-th proton and the probability that the i-th proton flies to the right in percentage points (-10^9 ≤ x_i ≤ 10^9, 1 ≤ v ≤ 10^6, 0 ≤ p_i ≤ 100). It is guaranteed that all x_i are distinct and sorted in increasing order.

Output

It's possible to prove that the answer can always be represented as a fraction P/Q, where P is an integer and Q is a natural number not divisible by 998 244 353. In this case, print P ⋅ Q^{-1} modulo 998 244 353.

Examples

Input


2
1 1 100
3 1 0


Output


1


Input


3
7 10 0
9 4 86
14 5 100


Output


0


Input


4
6 4 50
11 25 50
13 16 50
15 8 50


Output


150902884

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int read() {
  int X = 0, w = 1;
  char c = getchar();
  while (c < '0' || c > '9') {
    if (c == '-') w = -1;
    c = getchar();
  }
  while (c >= '0' && c <= '9') X = X * 10 + c - '0', c = getchar();
  return X * w;
}
const int N = 100000 + 10;
const int mod = 998244353;
int qpow(int a, int b) {
  int c = 1;
  for (; b; b >>= 1, a = 1ll * a * a % mod)
    if (b & 1) c = 1ll * c * a % mod;
  return c;
}
int n, x[N], v[N], p[2][N], cnt = 0;
struct node {
  int p, x, v, fl, fr;
} a[N << 1];
bool operator<(node a, node b) { return 1ll * a.x * b.v < 1ll * b.x * a.v; }
struct Matrix {
  int s[2][2];
  Matrix() { memset(s, 0, sizeof(s)); }
  int* operator[](int i) { return s[i]; }
};
Matrix operator*(Matrix a, Matrix b) {
  Matrix c;
  for (int k = 0; k < 2; ++k)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        c[i][j] = (c[i][j] + 1ll * a[i][k] * b[k][j]) % mod;
  return c;
}
Matrix mulv[N << 2];
void pushup(int o) { mulv[o] = mulv[(o << 1)] * mulv[(o << 1 | 1)]; }
void build(int o, int l, int r) {
  if (l == r) {
    mulv[o][0][0] = mulv[o][1][0] = p[0][l];
    mulv[o][0][1] = mulv[o][1][1] = p[1][l];
    return;
  }
  int mid = (l + r) >> 1;
  build((o << 1), l, mid), build((o << 1 | 1), mid + 1, r);
  pushup(o);
}
void modify(int o, int l, int r, int p, int fl, int fr) {
  if (l == r) {
    mulv[o][fl][fr] = 0;
    return;
  }
  int mid = (l + r) >> 1;
  if (p <= mid)
    modify((o << 1), l, mid, p, fl, fr);
  else
    modify((o << 1 | 1), mid + 1, r, p, fl, fr);
  pushup(o);
}
int main() {
  n = read();
  for (int i = 1; i <= n; ++i) {
    x[i] = read(), v[i] = read();
    p[1][i] = 1ll * read() * qpow(100, mod - 2) % mod;
    p[0][i] = (1 - p[1][i] + mod) % mod;
  }
  for (int i = 1; i < n; ++i) {
    a[++cnt] = (node){i + 1, x[i + 1] - x[i], v[i] + v[i + 1], 1, 0};
    if (v[i] < v[i + 1])
      a[++cnt] = (node){i + 1, x[i + 1] - x[i], v[i + 1] - v[i], 0, 0};
    if (v[i] > v[i + 1])
      a[++cnt] = (node){i + 1, x[i + 1] - x[i], v[i] - v[i + 1], 1, 1};
  }
  sort(a + 1, a + cnt + 1);
  build(1, 1, n);
  int ans = 0;
  for (int i = 1; i <= cnt; ++i) {
    int P = (mulv[1][0][0] + mulv[1][0][1]) % mod;
    modify(1, 1, n, a[i].p, a[i].fl, a[i].fr);
    int Q = (mulv[1][0][0] + mulv[1][0][1]) % mod;
    ans = (ans + 1ll * a[i].x * qpow(a[i].v, mod - 2) % mod * (P - Q + mod)) %
          mod;
  }
  printf(\"%d\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from fractions import Fraction
from bootcamp import Basebootcamp

MOD = 998244353

class Dlccbootcamp(Basebootcamp):
    def __init__(self, max_n=2):
        self.max_n = max_n  # 控制生成的管道数，默认为2
    
    def case_generator(self):
        # 生成较小的测试用例以确保计算可行
        n = random.randint(2, self.max_n)
        x = []
        current_x = random.randint(-100, 100)
        x.append(current_x)
        for _ in range(n-1):
            current_x += random.randint(1, 10)
            x.append(current_x)
        v = [random.randint(1, 10) for _ in range(n)]
        p = [random.choice([0, 100]) for _ in range(n)]  # 确保概率为0%或100%
        
        expected = self.compute_expected(x, v, p)
        return {
            'n': n,
            'pipes': list(zip(x, v, p)),
            'expected': expected
        }
    
    def compute_expected(self, x, v, p):
        directions = []
        for pi in p:
            if pi == 0:
                directions.append(0)  # 左
            else:
                directions.append(1)  # 右
        
        min_time = None
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                xi, xj = x[i], x[j]
                vi, vj = v[i], v[j]
                di, dj = directions[i], directions[j]
                
                # 计算碰撞时间
                if di == 1 and dj == 0:  # i右，j左
                    dx = xj - xi
                    dv = vi + vj
                    time = Fraction(dx, dv)
                elif di == 1 and dj == 1 and vi > vj:  # 同右i更快
                    dx = xj - xi
                    dv = vi - vj
                    time = Fraction(dx, dv)
                elif di == 0 and dj == 0 and vj > vi:  # 同左j更快
                    dx = xj - xi
                    dv = vj - vi
                    time = Fraction(dx, dv)
                else:
                    continue  # 无碰撞可能
                
                if min_time is None or time < min_time:
                    min_time = time
        
        if min_time is None:
            return 0
        else:
            P = min_time.numerator
            Q = min_time.denominator
            Q_inv = pow(Q, MOD-2, MOD)
            return (P * Q_inv) % MOD
    
    @staticmethod
    def prompt_func(question_case):
        pipes_desc = "\n".join(
            f"{x} {v} {p}" for x, v, p in question_case['pipes']
        )
        return f"""You are a physicist analyzing the Line Chillland Collider experiment. There are {question_case['n']} pipes emitting protons with given coordinates, speeds, and movement probabilities. 

Task:
Calculate the expected duration until the first proton collision. If no collision occurs, the duration is 0. Express the answer as P⋅Q⁻¹ modulo 998244353.

Input:
{question_case['n']}
{pipes_desc}

Format your answer as [answer]<result>[/answer], replacing <result> with the computed value. For example, use [answer]42[/answer] if the result is 42.

Provide the numerical answer within the tags."""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match) % MOD
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
