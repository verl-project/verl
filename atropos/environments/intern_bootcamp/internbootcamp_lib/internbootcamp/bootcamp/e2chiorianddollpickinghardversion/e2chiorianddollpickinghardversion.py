"""# 

### 谜题描述
This is the hard version of the problem. The only difference between easy and hard versions is the constraint of m. You can make hacks only if both versions are solved.

Chiori loves dolls and now she is going to decorate her bedroom!

<image>

As a doll collector, Chiori has got n dolls. The i-th doll has a non-negative integer value a_i (a_i < 2^m, m is given). Chiori wants to pick some (maybe zero) dolls for the decoration, so there are 2^n different picking ways.

Let x be the bitwise-xor-sum of values of dolls Chiori picks (in case Chiori picks no dolls x = 0). The value of this picking way is equal to the number of 1-bits in the binary representation of x. More formally, it is also equal to the number of indices 0 ≤ i < m, such that \left⌊ (x)/(2^i) \right⌋ is odd.

Tell her the number of picking ways with value i for each integer i from 0 to m. Due to the answers can be very huge, print them by modulo 998 244 353.

Input

The first line contains two integers n and m (1 ≤ n ≤ 2 ⋅ 10^5, 0 ≤ m ≤ 53) — the number of dolls and the maximum value of the picking way.

The second line contains n integers a_1, a_2, …, a_n (0 ≤ a_i < 2^m) — the values of dolls.

Output

Print m+1 integers p_0, p_1, …, p_m — p_i is equal to the number of picking ways with value i by modulo 998 244 353.

Examples

Input


4 4
3 5 8 14


Output


2 2 6 6 0 

Input


6 7
11 45 14 9 19 81


Output


1 2 11 20 15 10 5 0 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using LL = long long;
namespace _buff {
const size_t BUFF = 1 << 19;
char ibuf[BUFF], *ib = ibuf, *ie = ibuf;
char getc() {
  if (ib == ie) {
    ib = ibuf;
    ie = ibuf + fread(ibuf, 1, BUFF, stdin);
  }
  return ib == ie ? -1 : *ib++;
}
}  // namespace _buff
LL read() {
  using namespace _buff;
  LL ret = 0;
  bool pos = true;
  char c = getc();
  for (; (c < '0' || c > '9') && c != '-'; c = getc()) {
    assert(~c);
  }
  if (c == '-') {
    pos = false;
    c = getc();
  }
  for (; c >= '0' && c <= '9'; c = getc()) {
    ret = (ret << 3) + (ret << 1) + (c ^ 48);
  }
  return pos ? ret : -ret;
}
const int MOD = 998244353;
using uint = unsigned;
struct Z {
  uint v;
  Z(uint v = 0) : v(v) {}
  Z &operator+=(const Z &z) {
    v += z.v;
    if (v >= MOD) v -= MOD;
    return *this;
  }
  Z &operator-=(const Z &z) {
    if (v < z.v) v += MOD;
    v -= z.v;
    return *this;
  }
  Z &operator*=(const Z &z) {
    v = static_cast<uint64_t>(v) * z.v % MOD;
    return *this;
  }
};
ostream &operator<<(ostream &os, const Z &z) { return os << z.v; }
Z operator+(const Z &x, const Z &y) {
  return Z(x.v + y.v >= MOD ? x.v + y.v - MOD : x.v + y.v);
}
Z operator-(const Z &x, const Z &y) {
  return Z(x.v < y.v ? x.v + MOD - y.v : x.v - y.v);
}
Z operator*(const Z &x, const Z &y) {
  return Z(static_cast<uint64_t>(x.v) * y.v % MOD);
}
Z qpow(Z base, uint e) {
  Z ret(1);
  for (; e; e >>= 1) {
    if (e & 1) {
      ret *= base;
    }
    base *= base;
  }
  return ret;
}
const size_t L = 60;
using ull = uint64_t;
uint m;
struct LB {
  ull b[L];
  LB() { memset(b, 0, sizeof b); }
  void add(ull x) {
    for (uint i = m; i--;) {
      if (x >> i & 1) {
        if (b[i]) {
          x ^= b[i];
        } else {
          b[i] = x;
          for (uint j = i; j--;) {
            if (b[i] >> j & 1) b[i] ^= b[j];
          }
          for (uint j = i + 1; j < m; ++j) {
            if (b[j] >> i & 1) b[j] ^= b[i];
          }
          return;
        }
      }
    }
  }
};
ull b[L];
uint cnt;
int f[L];
void get_b(const LB &lb) {
  cnt = 0;
  for (uint i = 0; i < m; ++i) {
    if (lb.b[i]) {
      b[cnt++] = lb.b[i];
    }
  }
}
void dfs(ull cur = 0, uint i = 0) {
  if (i == cnt) {
    ++f[__builtin_popcountll(cur)];
    return;
  }
  dfs(cur, i + 1);
  dfs(cur ^ b[i], i + 1);
}
Z comb[L][L];
void prep_comb() {
  for (uint i = 0; i < L; ++i) {
    comb[i][0] = 1;
    for (uint j = 1; j <= i; ++j) {
      comb[i][j] = comb[i - 1][j] + comb[i - 1][j - 1];
    }
  }
}
int main() {
  int n = read();
  m = read();
  LB b;
  for (int i = 0; i < n; ++i) {
    b.add(read());
  }
  get_b(b);
  assert(cnt <= n);
  Z mul = qpow(2, n - cnt);
  if ((cnt << 1) <= m) {
    dfs();
    for (uint i = 0; i <= m; ++i) {
      cout << f[i] * mul << ' ';
    }
  } else {
    cnt = 0;
    for (uint i = 0; i < m; ++i) {
      ull cur = 1ull << i;
      for (uint j = 0; j < m; ++j) {
        cur ^= (b.b[j] >> i & 1) << j;
      }
      if (cur) {
        ::b[cnt++] = cur;
      }
    }
    dfs();
    mul *= qpow(2, MOD - 1 - cnt);
    prep_comb();
    for (uint i = 0; i <= m; ++i) {
      Z ans = 0;
      for (uint j = 0; j <= m; ++j) {
        for (uint k = 0; k <= i && k <= j; ++k) {
          Z cur = f[j] * comb[j][k] * comb[m - j][i - k];
          if (k & 1)
            ans -= cur;
          else
            ans += cur;
        }
      }
      cout << ans * mul << ' ';
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
from math import comb

MOD = 998244353

class E2chiorianddollpickinghardversionbootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=5):
        self.max_n = max_n
        self.max_m = max_m

    def case_generator(self):
        m = random.choice([0, 3, 4, 5]) if self.max_m >=5 else random.randint(0, self.max_m)
        n = random.randint(1, self.max_n)
        
        if m == 0:
            a_list = [0] * n
        else:
            a_list = [random.randint(0, (1 << m)-1) for _ in range(n)]
            # 确保有解的情况下至少保留一个非零元素
            if all(x == 0 for x in a_list):
                a_list[random.randint(0, n-1)] = random.randint(1, (1 << m)-1)
        
        expected_output = self.solve_case(n, m, a_list)
        return {
            'n': n,
            'm': m,
            'a': a_list,
            'expected_output': expected_output
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        a = question_case['a']
        problem = (
            f"## 问题描述\n"
            f"Chiori有{n}个人偶，每个人偶的值为{a}（每个值都小于2^{m}）。\n"
            f"需要计算所有子集的异或和的二进制表示中1的个数恰好为i的方案数（0 ≤ i ≤ {m}），结果模998244353。\n\n"
            f"## 输出格式\n"
            f"输出{m+1}个空格分隔的整数，分别对应i=0到i={m}的结果，放在[answer]标签内。\n"
            f"示例：\n[answer]1 0 2 3 0 0[/answer]"
        )
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return list(map(int, last_match.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_output']
        try:
            if len(solution) != len(expected):
                return False
            return all((s % MOD) == (e % MOD) for s, e in zip(solution, expected))
        except:
            return False

    @staticmethod
    def build_linear_basis(a_list, m):
        basis = [0] * m
        for x in a_list:
            if x == 0:
                continue
            for i in reversed(range(m)):  # 固定从高位到低位处理
                if (x >> i) & 1:
                    if basis[i]:
                        x ^= basis[i]
                    else:
                        basis[i] = x
                        # 消去低位
                        for j in reversed(range(i)):
                            if (basis[i] >> j) & 1:
                                basis[i] ^= basis[j]
                        # 消去高位
                        for j in range(i+1, m):
                            if (basis[j] >> i) & 1:
                                basis[j] ^= basis[i]
                        break
        non_zero = [b for b in basis if b != 0]
        return non_zero, basis

    @staticmethod
    def solve_case(n, m, a_list):
        if m == 0:
            return [pow(2, n, MOD)]

        non_zero, basis = E2chiorianddollpickinghardversionbootcamp.build_linear_basis(a_list, m)
        cnt = len(non_zero)
        pow2 = pow(2, n - cnt, MOD)
        result = [0]*(m+1)

        if 2 * cnt <= m:
            f = [0]*(m+1)
            
            def dfs(val, idx):
                if idx == cnt:
                    bits = bin(val).count('1')
                    if bits <= m:
                        f[bits] += 1
                    return
                dfs(val, idx+1)
                dfs(val ^ non_zero[idx], idx+1)
            
            dfs(0, 0)
            for i in range(m+1):
                result[i] = (f[i] * pow2) % MOD
        else:
            # 修正组合数计算逻辑
            comb_table = [[0]*(m+1) for _ in range(m+1)]
            for i in range(m+1):
                comb_table[i][0] = 1
                for j in range(1, i+1):
                    comb_table[i][j] = (comb_table[i-1][j] + comb_table[i-1][j-1]) % MOD

            # 构建对偶基
            new_b = []
            for i in range(m):
                cur = 1 << i
                for j in range(m):
                    if basis[j] and ((basis[j] >> i) & 1):
                        cur ^= 1 << j
                if cur != 0:
                    new_b.append(cur)
            
            dual_cnt = len(new_b)
            f = [0]*(m+1)
            
            def dfs_dual(val, idx):
                if idx == dual_cnt:
                    bits = bin(val).count('1')
                    if bits <= m:
                        f[bits] += 1
                    return
                dfs_dual(val, idx+1)
                dfs_dual(val ^ new_b[idx], idx+1)
            
            dfs_dual(0, 0)
            
            inv_pow = pow(2, dual_cnt, MOD)
            inv_pow = pow(inv_pow, MOD-2, MOD)
            total_mul = (pow2 * inv_pow) % MOD
            
            for i in range(m+1):
                res = 0
                for j in range(m+1):
                    if f[j] == 0:
                        continue
                    tmp = 0
                    for k in range(0, min(i, j)+1):
                        c = (comb_table[j][k] * comb_table[m-j][i-k]) % MOD
                        if k % 2 == 0:
                            tmp = (tmp + c) % MOD
                        else:
                            tmp = (tmp - c) % MOD
                    res = (res + f[j] * tmp) % MOD
                result[i] = (res * total_mul) % MOD

        return [x % MOD for x in result]
