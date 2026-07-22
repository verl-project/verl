"""# 

### 谜题描述
In mathematical terms, the sequence Fn of Fibonacci numbers is defined by the recurrence relation 

F1 = 1; F2 = 1; Fn = Fn - 1 + Fn - 2 (n > 2).

DZY loves Fibonacci numbers very much. Today DZY gives you an array consisting of n integers: a1, a2, ..., an. Moreover, there are m queries, each query has one of the two types:

  1. Format of the query \"1 l r\". In reply to the query, you need to add Fi - l + 1 to each element ai, where l ≤ i ≤ r. 
  2. Format of the query \"2 l r\". In reply to the query you should output the value of <image> modulo 1000000009 (109 + 9). 



Help DZY reply to all the queries.

Input

The first line of the input contains two integers n and m (1 ≤ n, m ≤ 300000). The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 109) — initial array a.

Then, m lines follow. A single line describes a single query in the format given in the statement. It is guaranteed that for each query inequality 1 ≤ l ≤ r ≤ n holds.

Output

For each query of the second type, print the value of the sum on a single line.

Examples

Input

4 4
1 2 3 4
1 1 4
2 1 4
1 2 4
2 1 3


Output

17
12

Note

After the first query, a = [2, 3, 5, 7].

For the second query, sum = 2 + 3 + 5 + 7 = 17.

After the third query, a = [2, 4, 6, 9].

For the fourth query, sum = 2 + 4 + 6 = 12.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 3e5 + 10;
const long long MOD = 1e9 + 9;
long long sum[N << 2], lazy1[N << 2], lazy2[N << 2], F[N];
inline void read(long long &x) {
  int f = 1;
  x = 0;
  char ch = getchar();
  while (ch < '0' || ch > '9') {
    if (ch == '-') f = -1;
    ch = getchar();
  }
  while (ch >= '0' && ch <= '9') {
    x = x * 10 + ch - '0';
    ch = getchar();
  }
  x = x * f;
}
struct Matrix {
  long long v[10][10];
  int lena, lenb;
  void init1(int a, int b) {
    v[0][0] = 1;
    v[0][1] = 1;
    v[1][0] = 1;
    v[1][1] = 0;
    lena = a, lenb = b;
  }
  void init2(int a, int b) {
    v[0][0] = 1;
    v[0][1] = 0;
    v[1][0] = 0;
    v[1][1] = 1;
    lena = a, lenb = b;
  }
  void init3(long long f1, long long f2, int a, int b) {
    v[0][0] = f2;
    v[0][1] = 0;
    v[1][0] = f1;
    v[1][1] = 0;
    lena = a, lenb = b;
  }
};
Matrix Mul(Matrix a, Matrix b) {
  Matrix ans;
  ans.lena = a.lena, ans.lenb = b.lenb;
  for (int i = 0; i < a.lena; i++) {
    for (int j = 0; j < b.lenb; j++) {
      ans.v[i][j] = 0;
      for (int k = 0; k < a.lenb; k++)
        ans.v[i][j] = (ans.v[i][j] + a.v[i][k] * b.v[k][j]) % MOD;
    }
  }
  return ans;
}
void print(Matrix a) {
  for (int i = 0; i < a.lena; i++) {
    for (int j = 0; j < a.lenb; j++) {
      cout << a.v[i][j] << ' ';
    }
    cout << endl;
  }
}
pair<long long, long long> get_f(long long f1, long long f2, int n) {
  pair<long long, long long> ans =
      pair<long long, long long>(f2, (f1 + f2) % MOD);
  if (n < 3) return ans;
  ans = pair<long long, long long>((f1 * F[n - 2] + f2 * F[n - 1]) % MOD,
                                   (f1 * F[n - 1] + f2 * F[n]) % MOD);
  return ans;
}
void modify(int rt, int l, int r, long long f1, long long f2) {
  lazy1[rt] = (lazy1[rt] + f1) % MOD;
  lazy2[rt] = (lazy2[rt] + f2) % MOD;
  sum[rt] = (sum[rt] + (get_f(f1, f2, r - l + 2).second - f2) + MOD) % MOD;
}
void pushDown(int rt, int l, int mid, int r) {
  if (lazy1[rt] && lazy2[rt]) {
    modify(rt << 1, l, mid, lazy1[rt], lazy2[rt]);
    pair<long long, long long> P = get_f(lazy1[rt], lazy2[rt], mid - l + 2);
    modify(rt << 1 | 1, mid + 1, r, P.first, P.second);
    lazy1[rt] = lazy2[rt] = 0;
  }
}
void Build(int rt, int l, int r) {
  lazy1[rt] = lazy2[rt] = sum[rt] = 0;
  if (l == r) {
    read(sum[rt]);
    return;
  }
  int mid = (l + r) >> 1;
  Build(rt << 1, l, mid);
  Build(rt << 1 | 1, mid + 1, r);
  sum[rt] = (sum[rt << 1] + sum[rt << 1 | 1]) % MOD;
}
void Update(int rt, int l, int r, int x, int y, long long f1, long long f2) {
  if (l == x && r == y) {
    modify(rt, l, r, f1, f2);
    return;
  }
  int mid = (l + r) >> 1;
  pushDown(rt, l, mid, r);
  if (y <= mid)
    Update(rt << 1, l, mid, x, y, f1, f2);
  else if (x > mid)
    Update(rt << 1 | 1, mid + 1, r, x, y, f1, f2);
  else {
    pair<long long, long long> P = get_f(f1, f2, mid - x + 2);
    Update(rt << 1, l, mid, x, mid, f1, f2),
        Update(rt << 1 | 1, mid + 1, r, mid + 1, y, P.first, P.second);
  }
  sum[rt] = (sum[rt << 1] + sum[rt << 1 | 1]) % MOD;
}
long long Query(int rt, int l, int r, int x, int y) {
  if (l == x && r == y) {
    return sum[rt];
  }
  int mid = (l + r) >> 1;
  pushDown(rt, l, mid, r);
  if (y <= mid)
    return Query(rt << 1, l, mid, x, y);
  else if (x > mid)
    return Query(rt << 1 | 1, mid + 1, r, x, y);
  else
    return (Query(rt << 1, l, mid, x, mid) +
            Query(rt << 1 | 1, mid + 1, r, mid + 1, y)) %
           MOD;
}
int main() {
  F[0] = 0, F[1] = 1;
  for (int i = 2; i < N; i++) F[i] = (F[i - 1] + F[i - 2]) % MOD;
  long long n, m, opt, l, r;
  read(n);
  read(m);
  Build(1, 1, n);
  for (int i = 1; i <= m; i++) {
    read(opt);
    read(l);
    read(r);
    if (opt == 1)
      Update(1, 1, n, l, r, 1, 1);
    else
      printf(\"%I64d\n\", Query(1, 1, n, l, r));
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

MOD = 10**9 + 9

class Cdzylovesfibonaccinumbersbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        参数设置：
        n: 数组长度（默认随机5-10）
        m: 查询数量（默认随机5-10）
        max_val: 初始数组最大值（默认1e9）
        """
        self.n = params.get('n', random.randint(5, 10))
        self.m = params.get('m', random.randint(5, 10))
        self.max_val = params.get('max_val', 10**9)
        self.fib_cache = {}
        self._precompute_fib(10**6)  # 预生成足够大的斐波那契数列

    def _precompute_fib(self, size):
        """预生成斐波那契数列备用"""
        self.fib_cache[1] = 1
        self.fib_cache[2] = 1
        for i in range(3, size+1):
            self.fib_cache[i] = (self.fib_cache[i-1] + self.fib_cache[i-2]) % MOD

    def case_generator(self):
        """动态生成测试用例并计算正确答案"""
        n, m = self.n, self.m
        
        # 生成初始数组
        arr = [random.randint(1, self.max_val) for _ in range(n)]
        
        queries = []
        answers = []
        sim_arr = arr.copy()
        
        # 生成查询序列（保证至少2个类型2查询）
        type2_count = 0
        for _ in range(m):
            # 确保最后两个查询是类型2（保证有答案）
            if _ >= m-2 and type2_count < 2:
                query_type = 2
            else:
                query_type = random.choices([1, 2], weights=[0.7, 0.3], k=1)[0]
            
            l = random.randint(1, n)
            r = random.randint(l, n)
            
            # 处理查询
            if query_type == 1:
                # 应用斐波那契叠加
                for i in range(l-1, r):
                    fib_num = (i - (l-1)) + 1  # F_{i-l+1} 的正确计算
                    sim_arr[i] = (sim_arr[i] + self.fib_cache[fib_num]) % MOD
            else:
                # 计算区间和
                current_sum = sum(sim_arr[l-1:r]) % MOD
                answers.append(current_sum)
                type2_count += 1
            
            queries.append({"type": query_type, "l": l, "r": r})

        return {
            "n": n,
            "m": m,
            "array": arr,
            "queries": queries,
            "answers": answers
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']}",
            ' '.join(map(str, question_case['array']))
        ]
        for q in question_case['queries']:
            input_lines.append(f"{q['type']} {q['l']} {q['r']}")

        problem = (
            "DZY给出了一个整数数组和m个查询，请处理以下两种操作：\n"
            "1. 类型1查询 (1 l r)：给区间[l, r]的每个元素a_i依次加上F_{i-l+1}（F为斐波那契数列，F₁=1, F₂=1）\n"
            "2. 类型2查询 (2 l r)：输出区间[l, r]元素和模1000000009\n\n"
            "输入格式：\n" +
            "\n".join(input_lines) +
            "\n\n请输出所有类型2查询的结果，每个结果占一行，并包裹在[answer]标签内，例如：\n"
            "[answer]\n17\n12\n[/answer]"
        )
        return problem

    @staticmethod
    def extract_output(output):
        import re
        answers = []
        # 匹配最后一个answer块内的所有数字
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if matches:
            last_match = matches[-1].strip()
            answers = [line.strip() for line in last_match.split('\n') if line.strip()]
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 将字符串答案转换为整数比较
            expected = identity['answers']
            return [int(ans) % MOD for ans in solution] == [x % MOD for x in expected]
        except (ValueError, KeyError, TypeError):
            return False
