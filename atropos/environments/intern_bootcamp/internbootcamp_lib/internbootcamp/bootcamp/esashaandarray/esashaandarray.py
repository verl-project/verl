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
inline int readInt() {
  static int n, ch;
  n = 0, ch = getchar();
  while (!isdigit(ch)) ch = getchar();
  while (isdigit(ch)) n = n * 10 + ch - '0', ch = getchar();
  return n;
}
const int MOD = 1000000007;
int t[2][2];
struct Matrix {
  int a[2][2];
  Matrix() { memset(a, 0, sizeof a); }
  int *operator[](int i) { return *(a + i); }
  const int *operator[](int i) const { return *(a + i); }
  inline void operator*=(const Matrix &m1) {
    memset(t, 0, sizeof t);
    for (int i = 0; i < 2; ++i)
      for (int k = 0; k < 2; ++k)
        for (int j = 0; j < 2; ++j)
          (t[i][j] += (long long)a[i][k] * m1.a[k][j] % MOD) %= MOD;
    memcpy(a, t, sizeof t);
  }
} I;
const int MAX_S = 64;
Matrix mem[MAX_S];
inline Matrix f(long long n) {
  Matrix A;
  A[0][0] = A[1][1] = 1;
  for (int i = 0; i < MAX_S; ++i)
    if ((n >> i) & 1) A *= mem[i];
  return A;
}
void init() {
  mem[0][0][0] = mem[0][0][1] = mem[0][1][0] = 1;
  for (int i = 1; i < MAX_S; ++i) {
    mem[i] = mem[i - 1];
    mem[i] *= mem[i - 1];
  }
}
const int MAX_N = 100000 + 3;
int n, m, a[MAX_N];
struct SegmentTree {
  static const int MAX_NODE = (1 << 17) * 2 + 1;
  struct Node {
    Matrix matrix, tagMul;
  } nodes[MAX_NODE];
  inline void multiply(int o, Matrix c) {
    Node &v = nodes[o];
    v.tagMul *= c;
    v.matrix *= c;
  }
  inline void pushDown(int o) {
    multiply(((o)*2 + 1), nodes[o].tagMul);
    multiply(((o)*2 + 2), nodes[o].tagMul);
    nodes[o].tagMul = I;
  }
  inline Matrix merge(const Matrix &a, const Matrix &b) {
    Matrix c;
    c[0][0] = (a[0][0] + b[0][0]) % MOD;
    c[0][1] = (a[0][1] + b[0][1]) % MOD;
    c[1][0] = (a[1][0] + b[1][0]) % MOD;
    c[1][1] = (a[1][1] + b[1][1]) % MOD;
    return c;
  }
  void build(int o, int l, int r, int a[]) {
    Node &v = nodes[o];
    v.tagMul = I;
    if (r - l == 1)
      v.matrix = f(a[l]);
    else {
      build(((o)*2 + 1), l, (((l) + (r)) >> 1), a),
          build(((o)*2 + 2), (((l) + (r)) >> 1), r, a);
      v.matrix = merge(nodes[((o)*2 + 1)].matrix, nodes[((o)*2 + 2)].matrix);
    }
  }
  void init(int n, int a[]) { build(0, 0, n, a); }
  Matrix query(int o, int l, int r, int a, int b) {
    if (l >= a && r <= b)
      return nodes[o].matrix;
    else {
      pushDown(o);
      Matrix res;
      if ((((l) + (r)) >> 1) > a)
        res = merge(res, query(((o)*2 + 1), l, (((l) + (r)) >> 1), a, b));
      if ((((l) + (r)) >> 1) < b)
        res = merge(res, query(((o)*2 + 2), (((l) + (r)) >> 1), r, a, b));
      return res;
    }
  }
  void modify(int o, int l, int r, int a, int b, Matrix x) {
    if (r <= a || l >= b) return;
    if (l >= a && r <= b)
      multiply(o, x);
    else {
      pushDown(o);
      if ((((l) + (r)) >> 1) > a)
        modify(((o)*2 + 1), l, (((l) + (r)) >> 1), a, b, x);
      if ((((l) + (r)) >> 1) < b)
        modify(((o)*2 + 2), (((l) + (r)) >> 1), r, a, b, x);
      nodes[o].matrix =
          merge(nodes[((o)*2 + 1)].matrix, nodes[((o)*2 + 2)].matrix);
    }
  }
  inline int query(int l, int r) { return query(0, 0, n, l, r)[1][0]; }
  inline void modify(int l, int r, int c) { modify(0, 0, n, l, r, f(c)); }
} segmentTree;
int main() {
  init();
  n = readInt(), m = readInt();
  for (int i = 0; i < n; ++i) a[i] = readInt();
  I[0][0] = 1, I[1][1] = 1;
  segmentTree.init(n, a);
  while (m--) {
    int type = readInt(), l = readInt() - 1, r = readInt();
    if (type == 1)
      segmentTree.modify(l, r, readInt());
    else if (type == 2)
      printf(\"%d\n\", segmentTree.query(l, r));
    else
      assert(false);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Esashaandarraybootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)
        self.m = params.get('m', 5)
        self.max_initial = params.get('max_initial', 5)
        self.max_x = params.get('max_x', 5)
    
    def case_generator(self):
        MOD = 10**9 +7
        
        # Generate initial array
        a_initial = [random.randint(1, self.max_initial) for _ in range(self.n)]
        queries = []
        diff = [0] * (self.n + 2)  # 差分数组(1-based)
        
        # Generate queries ensuring at least one type2
        type2_count = 0
        for _ in range(self.m):
            # Force last query to be type2 if no type2 generated
            if type2_count == 0 and len(queries) == self.m -1:
                t = 2
            else:
                t = random.choice([1,2])
            
            l = random.randint(1, self.n)
            r = random.randint(l, self.n)
            
            if t ==1:
                x = random.randint(1, self.max_x)
                queries.append({'type':1, 'l':l, 'r':r, 'x':x})
                # 关键修复点：移除原条件判断
                diff[l-1] += x
                diff[r] -= x  # 直接修改r位置，无需条件判断
            else:
                queries.append({'type':2, 'l':l, 'r':r})
                type2_count +=1
        
        # 计算delta数组的Prefix Sum
        delta = [0]*self.n
        current_diff =0
        for i in range(self.n):
            current_diff += diff[i]
            delta[i] = current_diff
        
        # 计算expected_outputs
        expected_outputs = []
        for q in queries:
            if q['type'] ==2:
                l, r = q['l'], q['r']
                total =0
                for i in range(l-1, r):
                    a = a_initial[i] + delta[i]
                    total = (total + self.fib_mod(a)) % MOD
                expected_outputs.append(total)
        
        # 确保至少一个type2
        if not expected_outputs:
            l, r = 1, self.n
            total = sum(self.fib_mod(a_initial[i] + delta[i]) for i in range(self.n)) % MOD
            expected_outputs.append(total)
            queries[-1] = {'type':2, 'l':l, 'r':r}
        
        return {
            'n': self.n,
            'm': self.m,
            'initial_array': a_initial,
            'queries': queries,
            'expected_outputs': expected_outputs
        }
    
    @staticmethod
    def fib_mod(x):
        MOD = 10**9 +7
        if x ==0:
            return 0
        if x ==1 or x ==2:
            return 1 % MOD
        
        def multiply(mat_a, mat_b):
            return [
                [(mat_a[0][0]*mat_b[0][0] + mat_a[0][1]*mat_b[1][0]) % MOD,
                 (mat_a[0][0]*mat_b[0][1] + mat_a[0][1]*mat_b[1][1]) % MOD],
                [(mat_a[1][0]*mat_b[0][0] + mat_a[1][1]*mat_b[1][0]) % MOD,
                 (mat_a[1][0]*mat_b[0][1] + mat_a[1][1]*mat_b[1][1]) % MOD]
            ]
        
        def matrix_power(mat, power):
            res = [[1,0],[0,1]]
            while power >0:
                if power %2 ==1:
                    res = multiply(res, mat)
                mat = multiply(mat, mat)
                power //=2
            return res
        
        trans = [[1,1],[1,0]]
        powered = matrix_power(trans, x-2)
        return (powered[0][0] + powered[0][1]) % MOD
    
    @staticmethod
    def prompt_func(question_case) -> str:
        problem_desc = "Sasha has an array of integers and needs to process several queries. Each query is of one of the following types:\n"
        problem_desc += "1. Type 1: Increase all elements from position l to r by x.\n"
        problem_desc += "2. Type 2: Compute the sum of Fibonacci numbers for elements from position l to r, modulo 1e9+7.\n"
        problem_desc += "The Fibonacci numbers are defined as f(1)=1, f(2)=1, f(x)=f(x-1)+f(x-2) for x>2.\n\n"
        problem_desc += f"Initial array: {question_case['initial_array']}\n\n"
        problem_desc += f"Queries (total {len(question_case['queries'])}):\n"
        for idx, q in enumerate(question_case['queries'], 1):
            if q['type'] ==1:
                problem_desc += f"{idx}. Type 1: l={q['l']}, r={q['r']}, x={q['x']}\n"
            else:
                problem_desc += f"{idx}. Type 2: l={q['l']}, r={q['r']}\n"
        problem_desc += "\nFor each Type 2 query, compute the sum of Fibonacci numbers modulo 1e9+7. Provide all answers in the order of the queries, each on a new line enclosed within [answer] and [/answer]. Example:\n[answer]\n5\n7\n9\n[/answer]"
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        solution = []
        for line in last_match.split('\n'):
            stripped = line.strip()
            if stripped:
                try:
                    solution.append(int(stripped))
                except:
                    pass
        return solution if solution else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('expected_outputs', [])
        return solution == expected
