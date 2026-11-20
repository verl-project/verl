"""# 

### 谜题描述
You are given an array a_1, a_2, …, a_n of integers. This array is non-increasing.

Let's consider a line with n shops. The shops are numbered with integers from 1 to n from left to right. The cost of a meal in the i-th shop is equal to a_i.

You should process q queries of two types:

  * 1 x y: for each shop 1 ≤ i ≤ x set a_{i} = max(a_{i}, y). 
  * 2 x y: let's consider a hungry man with y money. He visits the shops from x-th shop to n-th and if he can buy a meal in the current shop he buys one item of it. Find how many meals he will purchase. The man can buy a meal in the shop i if he has at least a_i money, and after it his money decreases by a_i. 

Input

The first line contains two integers n, q (1 ≤ n, q ≤ 2 ⋅ 10^5).

The second line contains n integers a_{1},a_{2}, …, a_{n} (1 ≤ a_{i} ≤ 10^9) — the costs of the meals. It is guaranteed, that a_1 ≥ a_2 ≥ … ≥ a_n.

Each of the next q lines contains three integers t, x, y (1 ≤ t ≤ 2, 1≤ x ≤ n, 1 ≤ y ≤ 10^9), each describing the next query.

It is guaranteed that there exists at least one query of type 2.

Output

For each query of type 2 output the answer on the new line.

Example

Input


10 6
10 10 10 6 6 5 5 5 3 1
2 3 50
2 4 10
1 3 10
2 2 36
1 4 7
2 2 17


Output


8
3
6
2

Note

In the first query a hungry man will buy meals in all shops from 3 to 10.

In the second query a hungry man will buy meals in shops 4, 9, and 10.

After the third query the array a_1, a_2, …, a_n of costs won't change and will be \{10, 10, 10, 6, 6, 5, 5, 5, 3, 1\}.

In the fourth query a hungry man will buy meals in shops 2, 3, 4, 5, 9, and 10.

After the fifth query the array a of costs will be \{10, 10, 10, 7, 6, 5, 5, 5, 3, 1\}.

In the sixth query a hungry man will buy meals in shops 2 and 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <class T>
using vc = vector<T>;
template <class T>
using vvc = vc<vc<T>>;
template <class T>
void mkuni(vector<T> &v) {
  sort(v.begin(), v.end());
  v.erase(unique(v.begin(), v.end()), v.end());
}
long long rand_int(long long l, long long r) {
  static mt19937_64 gen(chrono::steady_clock::now().time_since_epoch().count());
  return uniform_int_distribution<long long>(l, r)(gen);
}
template <class T>
void print(T x, int suc = 1) {
  cout << x;
  if (suc == 1)
    cout << '\n';
  else
    cout << ' ';
}
template <class T>
void print(const vector<T> &v, int suc = 1) {
  for (int i = 0; i < v.size(); i++)
    print(v[i], i == (int)(v.size()) - 1 ? suc : 2);
}
const int N = 3e5 + 10;
struct Tree {
  long long l, r, lazy, sum, mi, ma;
} tree[N << 2];
void push_up(int rt) {
  tree[rt].sum = tree[rt << 1].sum + tree[rt << 1 | 1].sum;
  tree[rt].ma = tree[rt << 1].ma;
  tree[rt].mi = tree[rt << 1 | 1].mi;
}
void build(int l, int r, int rt, vector<int> &a) {
  tree[rt].l = l, tree[rt].r = r, tree[rt].lazy = 0;
  if (l == r) {
    tree[rt].sum = tree[rt].mi = tree[rt].ma = a[l];
    return;
  }
  int mid = l + r >> 1;
  build(l, mid, rt << 1, a);
  build(mid + 1, r, rt << 1 | 1, a);
  push_up(rt);
}
void push_down(int rt) {
  if (tree[rt].lazy) {
    int x = tree[rt].lazy, l = tree[rt].l, r = tree[rt].r;
    tree[rt].lazy = 0;
    tree[rt << 1].sum = 1ll * (tree[rt << 1].r - tree[rt << 1].l + 1) * x;
    tree[rt << 1].mi = tree[rt << 1].ma = x;
    tree[rt << 1].lazy = x;
    tree[rt << 1 | 1].sum =
        1ll * (tree[rt << 1 | 1].r - tree[rt << 1 | 1].l + 1) * x;
    tree[rt << 1 | 1].mi = tree[rt << 1 | 1].ma = x;
    tree[rt << 1 | 1].lazy = x;
  }
}
void update_range(int L, int R, long long Y, int rt) {
  int l = tree[rt].l, r = tree[rt].r;
  if (tree[rt].mi >= Y || l > R) return;
  if (tree[rt].ma <= Y && r <= R) {
    tree[rt].sum = 1ll * (r - l + 1) * Y;
    tree[rt].mi = tree[rt].ma = Y;
    tree[rt].lazy = Y;
    return;
  }
  push_down(rt);
  update_range(L, R, Y, rt << 1);
  update_range(L, R, Y, rt << 1 | 1);
  push_up(rt);
}
int query_range(int L, int R, int rt, long long &Y) {
  int l = tree[rt].l, r = tree[rt].r;
  if (tree[rt].mi > Y || r < L) return 0;
  if (tree[rt].sum <= Y && l >= L) {
    Y -= tree[rt].sum;
    return r - l + 1;
  }
  push_down(rt);
  long long res = 0;
  res += query_range(L, R, rt << 1, Y);
  res += query_range(L, R, rt << 1 | 1, Y);
  return res;
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  int n, q;
  cin >> n >> q;
  vector<int> a(n + 1);
  for (int i = 1; i <= n; ++i) cin >> a[i];
  build(1, n, 1, a);
  while (q--) {
    long long x, y, op;
    cin >> op >> x >> y;
    if (op == 1) {
      update_range(1, x, y, 1);
    } else
      cout << query_range(x, n, 1, y) << '\n';
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from itertools import accumulate
from bootcamp import Basebootcamp

class Cgreedyshoppingbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 10)
        self.q = params.get('q', 5)
        self.n = max(1, self.n)  # 确保最小为1
        self.q = max(1, self.q)  # 确保至少一个查询

    def case_generator(self):
        # 生成非递增数组（优化版）
        a = [random.randint(1, 10**9)]
        for _ in range(1, self.n):
            a.append(random.randint(1, a[-1]))
        
        # 生成查询列表并确保类型2存在
        queries = []
        type2_indices = []
        for i in range(self.q):
            t = random.choices([1, 2], weights=[0.4, 0.6])[0]  # 增加类型2概率
            x = random.randint(1, self.n)
            y = random.randint(1, 10**9)
            queries.append([t, x, y])  # 统一使用列表存储
            if t == 2:
                type2_indices.append(i)

        # 确保至少一个类型2查询
        if not type2_indices:
            queries[-1] = [2, random.randint(1, self.n), random.randint(1, 10**9)]
            type2_indices = [self.q-1]

        # 预处理答案（优化模拟）
        current_a = a.copy()
        answers = []
        for op in queries:
            t, x, y = op
            if t == 1:
                # 使用二分查找确定有效更新范围
                left = 0
                right = x-1  # 转换为0-based索引
                update_pos = next((i for i in range(x) if current_a[i] < y), None)
                if update_pos is not None:
                    current_a[update_pos:x] = [max(y, val) for val in current_a[update_pos:x]]
            else:
                # 使用累积和优化计算
                prefix = list(accumulate(current_a[x-1:]))
                money = y
                count = 0
                for s in prefix:
                    if s > money:
                        break
                    count += 1
                    money -= s - (prefix[count-2] if count>1 else 0)
                answers.append(count)

        return {
            'n': self.n,
            'q': self.q,
            'initial_array': a,
            'queries': queries,  # 统一使用列表存储
            'answers': answers
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['q']}",
            ' '.join(map(str, question_case['initial_array']))
        ]
        for op in question_case['queries']:
            input_lines.append(f"{op[0]} {op[1]} {op[2]}")

        return f"""你正在处理餐馆消费查询系统，需要处理两种操作类型：

**规则详解**
1. 类型1 (1 x y)：将前x个餐馆的餐费更新为原值和y的较大值
2. 类型2 (2 x y)：顾客从第x个餐馆开始向后消费，直到余额不足

**输入格式**
{" ".join(input_lines[:2])}
{chr(10).join(input_lines[2:])}

**答案格式要求**
请将所有类型2查询的答案按顺序排列在[answer]标签内，例如：
[answer]
3
5
2
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answers = []
        for line in matches[-1].strip().split('\n'):
            if line.strip().isdigit():
                answers.append(int(line.strip()))
        return answers or None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answers']

# 验证示例
if __name__ == "__main__":
    bootcamp = Cgreedyshoppingbootcamp(n=10, q=6)
    case = bootcamp.case_generator()
    print("Generated Case:")
    print(f"Initial array: {case['initial_array']}")
    print(f"Queries: {case['queries']}")
    print(f"Expected answers: {case['answers']}")

    prompt = Cgreedyshoppingbootcamp.prompt_func(case)
    print("\nGenerated Prompt:")
    print(prompt)

    # 模拟模型回答
    response = f"[answer]\n" + "\n".join(map(str, case['answers'])) + "\n[/answer]"
    extracted = Cgreedyshoppingbootcamp.extract_output(response)
    print("\nExtracted Answer:", extracted)

    score = Cgreedyshoppingbootcamp.verify_score(response, case)
    print("Validation Score:", score)
