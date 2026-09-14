"""# 

### 谜题描述
Sasha has an array of integers a1, a2, ..., an. You have to perform m queries. There might be queries of two types:

  1. 1 l r x — increase all integers on the segment from l to r by values x; 
  2. 2 l r — find <image>, where f(x) is the x-th Fibonacci number. As this number may be large, you only have to find it modulo 109 + 7. 



In this problem we define Fibonacci numbers as follows: f(1) = 1, f(2) = 1, f(x) = f(x - 1) + f(x - 2) for all x > 2.

Sasha is a very talented boy and he managed to perform all queries in five seconds. Will you be able to write the program that performs as well as Sasha?

Input

The first line of the input contains two integers n and m (1 ≤ n ≤ 100 000, 1 ≤ m ≤ 100 000) — the number of elements in the array and the number of queries respectively.

The next line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 109).

Then follow m lines with queries descriptions. Each of them contains integers tpi, li, ri and may be xi (1 ≤ tpi ≤ 2, 1 ≤ li ≤ ri ≤ n, 1 ≤ xi ≤ 109). Here tpi = 1 corresponds to the queries of the first type and tpi corresponds to the queries of the second type.

It's guaranteed that the input will contains at least one query of the second type.

Output

For each query of the second type print the answer modulo 109 + 7.

Examples

Input

5 4
1 1 2 1 1
2 1 5
1 2 4 2
2 2 4
2 1 5


Output

5
7
9

Note

Initially, array a is equal to 1, 1, 2, 1, 1.

The answer for the first query of the second type is f(1) + f(1) + f(2) + f(1) + f(1) = 1 + 1 + 1 + 1 + 1 = 5. 

After the query 1 2 4 2 array a is equal to 1, 3, 4, 3, 1.

The answer for the second query of the second type is f(3) + f(4) + f(3) = 2 + 3 + 2 = 7.

The answer for the third query of the second type is f(1) + f(3) + f(4) + f(3) + f(1) = 1 + 2 + 3 + 2 + 1 = 9.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 123;
const int mod = 1e9 + 7;
const int INF = 1e9 + 1;
const long long INFL = 1e18 + 1;
const double eps = 1e-9;
const double pi = acos(-1.0);
inline void add(long long &a, long long b) {
  a += b;
  if (a >= mod) a -= mod;
}
inline long long sum(long long a, long long b) {
  add(a, b);
  return a;
}
inline long long mult(int a, int b) { return 1ll * a * b % mod; }
long long n, m, a[N];
struct matrix {
  long long a[5][5], n;
  matrix() {
    n = 2;
    memset(a, 0, sizeof a);
  }
  matrix(int _n) {
    n = _n;
    memset(a, 0, sizeof a);
  }
  matrix operator*(matrix B) {
    matrix C;
    for (int i = 1; i <= 2; i++) {
      for (int j = 1; j <= 2; j++) {
        for (int k = 1; k <= 2; k++) {
          C.a[i][j] += a[i][k] * B.a[k][j];
        }
      }
    }
    for (int i = 1; i <= 2; i++)
      for (int j = 1; j <= 2; j++) C.a[i][j] %= mod;
    return C;
  }
  matrix operator+(matrix B) {
    matrix C;
    for (int i = 1; i <= 2; i++) {
      for (int j = 1; j <= 2; j++) {
        for (int k = 1; k <= 2; k++) {
          C.a[i][j] = sum(a[i][k], B.a[k][j]);
        }
      }
    }
    return C;
  }
  inline void mode() {
    for (int i = 1; i <= 2; i++)
      for (int j = 1; j <= 2; j++) a[i][j] %= mod;
  }
  inline void erase() { memset(a, 0, sizeof a); }
} B;
inline void paint(matrix &B) {
  B.a[1][1] = 0;
  B.a[2][1] = 1;
  B.a[1][2] = 1;
  B.a[2][2] = 1;
}
inline void make(matrix &B) {
  B.a[1][2] = 0;
  B.a[2][1] = 0;
  B.a[1][1] = 1;
  B.a[2][2] = 1;
}
struct node {
  long long x, y;
  matrix to;
  bool need;
  node() {
    x = y = 0;
    need = 0;
    make(to);
  }
} T[4 * N];
inline matrix binpow(matrix B, long long n) {
  matrix ans;
  ans.erase();
  ans.a[1][1] = 1;
  ans.a[1][2] = 1;
  while (n) {
    if (n & 1) ans = ans * B;
    B = B * B;
    n >>= 1;
  }
  return ans;
}
inline matrix Mypow(matrix B, long long n) {
  matrix ans;
  paint(ans);
  while (n) {
    if (n & 1) ans = ans * B;
    B = B * B;
    n >>= 1;
  }
  ans.mode();
  return ans;
}
inline void build(int v = 1, int tl = 1, int tr = n) {
  if (tl == tr) {
    matrix aa = binpow(B, a[tl] - 1);
    matrix bb = binpow(B, a[tl]);
    T[v].x = aa.a[1][1];
    T[v].y = bb.a[1][1];
  } else {
    int tm = (tl + tr) >> 1;
    build(v + v, tl, tm);
    build(v + v + 1, tm + 1, tr);
    T[v].x = sum(T[v + v].x, T[v + v + 1].x);
    T[v].y = sum(T[v + v].y, T[v + v + 1].y);
  }
}
inline void push(int v, int tl, int tr) {
  if (!T[v].need) return;
  matrix too;
  too.erase();
  too.a[1][1] = T[v].x;
  too.a[1][2] = T[v].y;
  too = too * T[v].to;
  T[v].x = too.a[1][1];
  T[v].y = too.a[1][2];
  if (tr != tl) {
    T[v + v].to = T[v + v].to * T[v].to;
    T[v + v + 1].to = T[v + v + 1].to * T[v].to;
    T[v + v].need = T[v + v + 1].need = 1;
  }
  T[v].to.erase();
  make(T[v].to);
  T[v].need = 0;
}
inline void upd(int l, int r, matrix X, int v = 1, int tl = 1, int tr = n) {
  push(v, tl, tr);
  if (tl > r || tr < l) return;
  if (l <= tl && tr <= r) {
    T[v].to = T[v].to * X;
    T[v].need = 1;
    push(v, tl, tr);
    return;
  }
  int tm = (tl + tr) >> 1;
  upd(l, r, X, v + v, tl, tm);
  upd(l, r, X, v + v + 1, tm + 1, tr);
  T[v].x = sum(T[v + v].x, T[v + v + 1].x);
  T[v].y = sum(T[v + v].y, T[v + v + 1].y);
}
inline long long get(int l, int r, int v = 1, int tl = 1, int tr = n) {
  push(v, tl, tr);
  if (tl > r || tr < l) return 0ll;
  if (l <= tl && tr <= r) return T[v].x;
  int tm = (tl + tr) >> 1;
  return sum(get(l, r, v + v, tl, tm), get(l, r, v + v + 1, tm + 1, tr));
}
int main() {
  scanf(\"%I64d%I64d\", &n, &m);
  for (int i = 1; i <= n; i++) scanf(\"%I64d\", &a[i]);
  paint(B);
  build();
  while (m--) {
    long long tp, l, r, x;
    scanf(\"%I64d\", &tp);
    if (tp == 1) {
      scanf(\"%I64d%I64d%I64d\", &l, &r, &x);
      matrix X = Mypow(B, x - 1);
      upd(l, r, X);
    } else {
      scanf(\"%I64d%I64d\", &l, &r);
      printf(\"%I64d\n\", get(l, r));
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from typing import List
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Csashaandarraybootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=5, max_x=3):
        self.max_n = max_n
        self.max_m = max_m
        self.max_x = max_x
        self.fib_cache = {}
        self._precompute_fib(max_x * (max_m + 2))  # 扩展预计算范围
        super().__init__()

    def _precompute_fib(self, size):
        """动态扩展的斐波那契预计算"""
        if size < 1: return
        self.fib_cache = {1: 1, 2: 1}
        a, b = 1, 1
        for i in range(3, size+1):
            a, b = b, (a + b) % MOD
            self.fib_cache[i] = b

    def case_generator(self):
        """健壮的测试用例生成"""
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        
        # 生成查询序列
        queries = []
        type2_indices = []
        for i in range(m):
            if len(type2_indices) == 0 and i == m-1:
                queries.append(('query', 1, n))
                type2_indices.append(i)
            else:
                if random.random() < 0.3:  # 30%概率生成查询
                    l = random.randint(1, n)
                    r = random.randint(l, n)
                    queries.append(('query', l, r))
                    type2_indices.append(i)
                else:
                    l = random.randint(1, n)
                    r = random.randint(l, n)
                    x = random.randint(1, self.max_x)
                    queries.append(('update', l, r, x))
        
        # 确保至少一个查询
        if not type2_indices:
            l = random.randint(1, n)
            r = random.randint(l, n)
            queries[-1] = ('query', l, r)
            type2_indices.append(len(queries)-1)

        # 生成初始数组
        arr = [random.randint(1, self.max_x) for _ in range(n)]
        current = arr.copy()
        
        # 构建输入和预期输出
        input_lines = [
            f"{n} {m}",
            ' '.join(map(str, arr))
        ]
        expected = []
        
        for q in queries:
            if q[0] == 'update':
                _, l, r, x = q
                input_lines.append(f"1 {l} {r} {x}")
                for i in range(l-1, r):
                    current[i] += x
                    # 动态扩展预计算缓存
                    if current[i] > len(self.fib_cache):
                        self._precompute_fib(current[i] * 2)
            else:
                _, l, r = q
                input_lines.append(f"2 {l} {r}")
                total = 0
                for i in range(l-1, r):
                    total = (total + self.fib_cache.get(current[i], 0)) % MOD
                expected.append(total)

        return {
            'input': '\n'.join(input_lines),
            'expected': expected,
            '_state': current  # 用于验证的完整状态
        }

    @staticmethod
    def prompt_func(case) -> str:
        return f"""根据输入处理数组更新和查询：
{case['input']}

每个查询结果按顺序用[ANSWER]标签包裹，例如：
[ANSWER]
42
[/ANSWER]"""

    @staticmethod
    def extract_output(text: str) -> List[int]:
        last_answer = re.findall(r'\[ANSWER\](.*?)\[\/ANSWER\]', text, re.DOTALL)
        if not last_answer:
            return None
        numbers = []
        for line in last_answer[-1].splitlines():
            line = line.strip()
            if line and line.isdigit():
                numbers.append(int(line))
        return numbers or None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
