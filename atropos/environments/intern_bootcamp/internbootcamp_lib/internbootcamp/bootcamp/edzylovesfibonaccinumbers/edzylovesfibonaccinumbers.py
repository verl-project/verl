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
const long long INF = 1e18 + 5;
const int mod = 1e9 + 9;
const int N = 3 * 1e5 + 5;
namespace FIB {
struct Matrix {
  int a[3][3];
  Matrix() { memset(a, 0, sizeof(a)); }
  Matrix operator*(const Matrix& M2) {
    Matrix result;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
          result.a[i][j] =
              (result.a[i][j] + 1ll * this->a[i][k] * M2.a[k][j]) % mod;
        }
    return result;
  }
  static Matrix eye() {
    Matrix res;
    for (int i = 0; i < 3; i++) res.a[i][i] = 1;
    return res;
  }
  Matrix exp(int pw) {
    if (pw == 0) return eye();
    if (pw == 1) return *this;
    Matrix half = this->exp(pw / 2);
    if (pw % 2) return half * half * (*this);
    return half * half;
  }
};
Matrix initMat() {
  Matrix res;
  int dx[] = {0, 0, 1, 2, 2};
  int dy[] = {0, 1, 0, 0, 2};
  for (int i = 0; i < 5; i++) res.a[dx[i]][dy[i]] = 1;
  return res;
}
Matrix matrices[N];
void initMatrices() {
  matrices[0] = Matrix::eye();
  matrices[1] = initMat();
  for (int i = 2; i < N; i++) matrices[i] = matrices[i - 1] * matrices[1];
}
pair<int, pair<int, int> > fib(int a, int b, int n) {
  Matrix& base = matrices[n - 1];
  pair<int, pair<int, int> > res;
  res.first = (1ll * base.a[0][0] * b + 1ll * base.a[0][1] * a +
               1ll * base.a[0][2] * a) %
              mod;
  res.second.first = (1ll * base.a[1][0] * b + 1ll * base.a[1][1] * a +
                      1ll * base.a[1][2] * a) %
                     mod;
  res.second.second = (1ll * base.a[2][0] * b + 1ll * base.a[2][1] * a +
                       1ll * base.a[2][2] * a) %
                      mod;
  return res;
}
}  // namespace FIB
namespace SegTree {
int n;
int s[4 * N], a[4 * N], lazyA[4 * N], lazyB[4 * N];
inline int aggr(int val1, int val2) { return (val1 + val2) % mod; }
void build(int l, int r, int id) {
  if (l == r) {
    s[id] = a[l];
  } else {
    int mid = (l + r) / 2;
    build(l, mid, id * 2);
    build(mid + 1, r, id * 2 + 1);
    s[id] = aggr(s[id * 2], s[id * 2 + 1]);
  }
}
void upd(int id, int l, int r, int va, int vb) {
  if (l == r) {
    s[id] = aggr(s[id], va);
  } else {
    lazyA[id] = aggr(lazyA[id], va);
    lazyB[id] = aggr(lazyB[id], vb);
    s[id] = aggr(s[id], FIB::fib(va, vb, r - l + 1).second.second);
  }
}
void shift(int id, int l, int r) {
  if (lazyA[id]) {
    int mid = (l + r) / 2;
    int va = lazyA[id];
    int vb = lazyB[id];
    upd(id * 2, l, mid, va, vb);
    pair<int, pair<int, int> > rb = FIB::fib(va, vb, mid + 2 - l);
    upd(id * 2 + 1, mid + 1, r, rb.second.first, rb.first);
    lazyA[id] = 0;
    lazyB[id] = 0;
  }
}
void update(int x, int y, int l, int r, int id, int va, int vb) {
  if (l >= x && r <= y) {
    pair<int, pair<int, int> > rb = FIB::fib(va, vb, l - x + 1);
    upd(id, l, r, rb.second.first, rb.first);
    return;
  }
  if (y < l || x > r) return;
  shift(id, l, r);
  int mid = (l + r) / 2;
  update(x, y, l, mid, id * 2, va, vb);
  update(x, y, mid + 1, r, id * 2 + 1, va, vb);
  s[id] = aggr(s[id * 2], s[id * 2 + 1]);
}
int query(int x, int y, int l, int r, int id) {
  if (l >= x && r <= y) {
    return s[id];
  }
  if (y < l || x > r) return 0;
  shift(id, l, r);
  int mid = (l + r) / 2;
  return aggr(query(x, y, l, mid, id * 2), query(x, y, mid + 1, r, id * 2 + 1));
}
}  // namespace SegTree
int main() {
  ios_base::sync_with_stdio(false);
  FIB::initMatrices();
  int n, m;
  cin >> n >> m;
  SegTree::n = n;
  for (int i = 0; i < n; i++) cin >> SegTree::a[i];
  SegTree::build(0, n - 1, 1);
  for (int i = 0; i < m; i++) {
    int choice, a, b;
    cin >> choice >> a >> b;
    a -= 1;
    b -= 1;
    if (choice == 1) {
      SegTree::update(a, b, 0, n - 1, 1, 1, 1);
    } else {
      int ans = SegTree::query(a, b, 0, n - 1, 1);
      cout << ans << endl;
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

MOD = 10**9 + 9

class Edzylovesfibonaccinumbersbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_range = params.get('n_range', (1, 5))
        self.m_range = params.get('m_range', (1, 5))
        self.a_range = params.get('a_range', (1, 10))
    
    def case_generator(self):
        n = random.randint(*self.n_range)
        m = random.randint(*self.m_range)
        a = [random.randint(*self.a_range) for _ in range(n)]
        queries = []
        has_type2 = False
        
        # Generate query sequence with at least one type2
        for _ in range(m):
            if has_type2:
                query_type = random.choices([1, 2], weights=[0.7, 0.3], k=1)[0]
            else:
                query_type = random.choice([1, 2])
            
            l = random.randint(1, n)
            r = random.randint(l, n)
            queries.append((query_type, l, r))
            if query_type == 2:
                has_type2 = True
        
        # Force last query to be type2 if no type2 exists
        if not has_type2 and queries:
            queries[-1] = (2, queries[-1][1], queries[-1][2])
            has_type2 = True
        
        # Precompute Fibonacci numbers
        fib_cache = [0, 1, 1]
        def get_fib(k):
            while len(fib_cache) <= k:
                fib_cache.append((fib_cache[-1] + fib_cache[-2]) % MOD)
            return fib_cache[k]
        
        # Simulate query processing
        current_a = a.copy()
        expected_outputs = []
        for q in queries:
            q_type, l, r = q
            l_idx = l - 1
            r_idx = r - 1
            if q_type == 1:
                for i in range(l_idx, r_idx + 1):
                    fib_index = (i - l_idx) + 1  # F_1 to F_{r-l+1}
                    get_fib(fib_index)  # Ensure cache
                    current_a[i] = (current_a[i] + fib_cache[fib_index]) % MOD
            else:
                segment_sum = sum(current_a[l_idx:r_idx+1]) % MOD
                expected_outputs.append(segment_sum)
        
        return {
            'initial_array': a,
            'queries': queries,
            'expected_outputs': expected_outputs
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [
            f"{len(question_case['initial_array'])} {len(question_case['queries'])}",
            ' '.join(map(str, question_case['initial_array']))
        ]
        for q in question_case['queries']:
            input_lines.append(f"{q[0]} {q[1]} {q[2]}")
        input_txt = '\n'.join(input_lines)
        
        return f"""## Fibonacci Query Puzzle ##
You are given:
1. An array of integers
2. A series of update and query operations

For UPDATE operations (type 1):
- Add Fibonacci numbers to a range: "1 l r"
- Fibonacci sequence is defined as F₁=1, F₂=1, Fₙ=Fₙ₋₁+Fₙ₋₂
- Specifically, for each i in [l, r], add F(i-l+1) to the element

For QUERY operations (type 2):
- Calculate sum of a range modulo {MOD}: "2 l r"

Input Format:
{input_txt}

Output Requirements:
- For each type 2 query, output the sum on a new line
- Enclose all results between [answer] and [/answer]
Example:
[answer]
42
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        # Find all possible answer blocks
        matches = list(re.finditer(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL))
        if not matches:
            return None
        
        # Extract last answer block
        last_match = matches[-1]
        content = last_match.group(1).strip()
        
        # Parse numerical results
        results = []
        for line in content.split('\n'):
            cleaned = line.strip()
            if cleaned:
                try:
                    results.append(int(cleaned))
                except ValueError:
                    pass
        
        return results if results else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('expected_outputs', [])

# 关键修正点说明
"""
1. **答案提取逻辑增强**:
   - 改用`re.finditer`查找所有匹配的答案块
   - 显式选择最后一个匹配的答案块进行处理
   - 增强容错处理：跳过无法转换的数字行

2. **斐波那契数缓存优化**:
   - 添加动态生成斐波那契数的缓存机制
   - 确保大索引值时的计算效率

3. **问题描述标准化**:
   - 添加更清晰的规则说明和格式示例
   - 强调模数要求和答案包围标签

4. **边界条件处理**:
   - 强制保证至少存在一个类型2查询
   - 严格校验数组索引范围
"""
