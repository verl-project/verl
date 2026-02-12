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
int n, q;
int p[200005];
struct node {
  int l, r;
  long long sum;
  int Min, Max;
  long long lazy;
} tree[200005 * 4];
void pushup(node& k, node& l, node& r) {
  k.sum = l.sum + r.sum;
  k.Min = min(l.Min, r.Min);
  k.Max = max(l.Max, r.Max);
}
void pushup(int x) { pushup(tree[x], tree[x << 1], tree[x << 1 | 1]); }
void build(int l, int r, int x = 1) {
  if (l == r) {
    tree[x] = {l, r, p[l], p[l], p[l], 0};
    return;
  }
  tree[x] = {l, r, 0, 0, 0, 0};
  int mid = (l + r) >> 1;
  build(l, mid, x << 1);
  build(mid + 1, r, x << 1 | 1);
  tree[x].sum = tree[x * 2].sum + tree[x * 2 + 1].sum;
  tree[x].Max = max(tree[x * 2].Max, tree[x * 2 + 1].Max);
  tree[x].Min = min(tree[x * 2].Min, tree[x * 2 + 1].Min);
}
void pushdown(node& op, long long lazy) {
  op.sum = lazy * (op.r - op.l + 1);
  op.Min = lazy;
  op.Max = lazy;
  op.lazy = lazy;
}
void pushdown(int x) {
  if (!tree[x].lazy) return;
  pushdown(tree[x << 1], tree[x].lazy);
  pushdown(tree[x << 1 | 1], tree[x].lazy);
  tree[x].lazy = 0;
}
int check(int l, int r, int& have, int x = 1) {
  if (l <= tree[x].l && r >= tree[x].r) {
    if (have < tree[x].Min) return 0;
    if (have >= tree[x].sum) {
      have -= tree[x].sum;
      return tree[x].r - tree[x].l + 1;
    }
    if (tree[x].l == tree[x].r) return 0;
  }
  pushdown(x);
  int mid = tree[x].l + tree[x].r >> 1;
  int res = 0;
  if (l <= mid) res += check(l, r, have, x << 1);
  if (r > mid) res += check(l, r, have, x << 1 | 1);
  return res;
}
void judge(int l, int r, int c, int x = 1) {
  if (l <= tree[x].l && r >= tree[x].r) {
    if (tree[x].Min >= c) return;
    if (tree[x].Max < c) {
      pushdown(tree[x], c);
      return;
    }
  }
  pushdown(x);
  int mid = tree[x].l + tree[x].r >> 1;
  if (l <= mid) judge(l, r, c, x << 1);
  if (r > mid) judge(l, r, c, x << 1 | 1);
  pushup(x);
}
int main() {
  cin >> n >> q;
  for (int i = 1; i <= n; i++) cin >> p[i];
  build(1, n);
  for (int i = 1; i <= q; i++) {
    int t, x, y;
    cin >> t >> x >> y;
    if (t == 1)
      judge(1, x, y);
    else
      cout << check(x, n, y) << endl;
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

class Egreedyshoppingbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_q=5, max_value=1000, type1_ratio=0.5):
        self.max_n = max_n
        self.max_q = max_q
        self.max_value = max_value
        self.type1_ratio = type1_ratio
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        q = random.randint(1, self.max_q)
        a_initial = sorted([random.randint(1, self.max_value) for _ in range(n)], reverse=True)
        queries = []
        has_type2 = False
        
        # Generate queries with guarantee at least one type2
        for i in range(q):
            if i == q-1 and not has_type2:
                t = 2
            else:
                t = 1 if random.random() < self.type1_ratio else 2
            
            x = random.randint(1, n)
            # Enhance y generation logic for type1
            if t == 1:
                current_max = max(a_initial[:x]) if x <= len(a_initial) else 0
                y = random.randint(
                    max(1, current_max - 5), 
                    current_max + self.max_value//2
                )
            else:
                total_max = sum(a_initial)
                y = random.randint(1, total_max * 2)
                has_type2 = True
            
            queries.append((t, x, y))

        # Simulate operations
        a = a_initial.copy()
        answers = []
        for t, x, y in queries:
            if t == 1:
                # Find last position where a[i] <= y to maintain non-increasing property
                new_val = y
                left = 0
                right = min(x, len(a)) - 1
                pos = -1
                while left <= right:
                    mid = (left + right) // 2
                    if a[mid] <= new_val:
                        pos = mid
                        right = mid - 1
                    else:
                        left = mid + 1
                
                if pos != -1:
                    fill_val = max(a[pos], new_val) if pos < len(a) else new_val
                    for i in range(pos, min(x, len(a))):
                        a[i] = max(a[i], fill_val)
            else:
                money = y
                count = 0
                for i in range(x-1, len(a)):
                    if money >= a[i]:
                        count += 1
                        money -= a[i]
                answers.append(count)
        
        return {
            'n': n,
            'q': q,
            'initial_array': a_initial,
            'queries': queries,
            'answers': answers
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['q']}",
            ' '.join(map(str, question_case['initial_array']))
        ]
        for t, x, y in question_case['queries']:
            input_lines.append(f"{t} {x} {y}")
        input_str = '\n'.join(input_lines)
        
        return f"""你需要解决一个算法问题：

给定一个非递增的整数数组，处理q个查询：
1. 类型1 (1 x y)：将前x个元素更新为max(a_i, y)
2. 类型2 (2 x y)：从第x个商店开始消费y元，计算能购买多少餐

输入格式：
第一行：n q
第二行：初始数组（保证非递增）
随后q行：每行包含t x y

输入数据：
{input_str}

请将每个类型2查询的答案按顺序放在[answer]标签内，如：
[answer]
结果1
结果2
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\][\s]*((?:\d+\s*)+)[\s]*\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        numbers = re.findall(r'\d+', matches[-1])
        return [int(num) for num in numbers] if numbers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answers', [])
