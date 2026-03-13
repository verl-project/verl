"""# 

### 谜题描述
You are given circular array a0, a1, ..., an - 1. There are two types of operations with it: 

  * inc(lf, rg, v) — this operation increases each element on the segment [lf, rg] (inclusively) by v; 
  * rmq(lf, rg) — this operation returns minimal value on the segment [lf, rg] (inclusively). 



Assume segments to be circular, so if n = 5 and lf = 3, rg = 1, it means the index sequence: 3, 4, 0, 1.

Write program to process given sequence of operations.

Input

The first line contains integer n (1 ≤ n ≤ 200000). The next line contains initial state of the array: a0, a1, ..., an - 1 ( - 106 ≤ ai ≤ 106), ai are integer. The third line contains integer m (0 ≤ m ≤ 200000), m — the number of operartons. Next m lines contain one operation each. If line contains two integer lf, rg (0 ≤ lf, rg ≤ n - 1) it means rmq operation, it contains three integers lf, rg, v (0 ≤ lf, rg ≤ n - 1; - 106 ≤ v ≤ 106) — inc operation.

Output

For each rmq operation write result for it. Please, do not use %lld specificator to read or write 64-bit integers in C++. It is preffered to use cout (also you may use %I64d).

Examples

Input

4
1 2 3 4
4
3 0
3 0 -1
0 1
2 1


Output

1
0
0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 5;
struct segNode {
  long long val;
  long long lazy;
  int l, r;
  segNode() { val = lazy = 0; }
  segNode(int a, int b, int c) {
    val = a;
    l = b;
    r = c;
    lazy = 0;
  }
};
int arr[N];
segNode segTree[4 * N];
void shift(int num, int left, int right) {
  int upd = segTree[num].lazy;
  segTree[num].val += upd;
  if (segTree[num].l != segTree[num].r) {
    segTree[left].lazy += upd;
    segTree[right].lazy += upd;
  }
  segTree[num].lazy = 0;
}
void build(int l, int r, int num) {
  if (l > r) return;
  if (l == r) {
    segTree[num] = segNode(arr[l], l, r);
    return;
  }
  int mid = (l + r) >> 1, left = num << 1, right = left | 1;
  build(l, mid, left);
  build(mid + 1, r, right);
  segTree[num] = segNode(min(segTree[left].val, segTree[right].val), l, r);
}
void update(int l, int r, int num, int lu, int ru, int inc) {
  if (l > r) return;
  int mid = (l + r) >> 1, left = num << 1, right = left | 1;
  if (segTree[num].lazy != 0) shift(num, left, right);
  if (l > ru || r < lu) return;
  if (l >= lu && r <= ru) {
    segTree[num].lazy += inc;
    shift(num, left, right);
    return;
  }
  update(l, mid, left, lu, ru, inc);
  update(mid + 1, r, right, lu, ru, inc);
  segTree[num].val = min(segTree[right].val, segTree[left].val);
}
long long query(int l, int r, int num, int lq, int rq) {
  if (l > r) return (long long)1e17;
  int mid = (l + r) >> 1, left = num << 1, right = left | 1;
  if (segTree[num].lazy != 0) shift(num, left, right);
  if (l > rq || r < lq) return (long long)1e17;
  if (l >= lq && r <= rq) {
    return segTree[num].val;
  }
  return min(query(l, mid, left, lq, rq), query(mid + 1, r, right, lq, rq));
}
int main() {
  int n;
  scanf(\"%d\", &n);
  for (int i = 0; i < n; ++i) {
    scanf(\"%d\", arr + i);
  }
  build(0, n - 1, 1);
  int q;
  scanf(\"%d\", &q);
  while (q--) {
    int x, y;
    char ch;
    scanf(\"%d%d%c\", &x, &y, &ch);
    if (ch == ' ') {
      int v;
      scanf(\"%d\", &v);
      if (x <= y)
        update(0, n - 1, 1, x, y, v);
      else
        update(0, n - 1, 1, x, n - 1, v), update(0, n - 1, 1, 0, y, v);
    } else {
      long long ans;
      if (x <= y)
        ans = query(0, n - 1, 1, x, y);
      else
        ans = min(query(0, n - 1, 1, x, n - 1), query(0, n - 1, 1, 0, y));
      printf(\"%lld\n\", ans);
    }
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ccircularrmqbootcamp(Basebootcamp):
    def __init__(self, n_range=(4, 10), m_range=(3, 8), **params):
        self.n_range = n_range
        self.m_range = m_range
        super().__init__(**params)
    
    def case_generator(self):
        n = random.randint(*self.n_range)
        array = [random.randint(-10, 10) for _ in range(n)]
        m = random.randint(*self.m_range)
        operations = []
        expected_outputs = []
        current_array = array.copy()
        
        for _ in range(m):
            if random.random() < 0.5:
                # Generate inc operation
                lf = random.randint(0, n-1)
                rg = random.randint(0, n-1)
                v = random.randint(-5, 5)
                operations.append({'type': 'inc', 'lf': lf, 'rg': rg, 'v': v})
                
                # Apply inc to current_array
                if lf <= rg:
                    for i in range(lf, rg+1):
                        current_array[i] += v
                else:
                    for i in range(lf, n):
                        current_array[i] += v
                    for i in range(0, rg+1):
                        current_array[i] += v
            else:
                # Generate rmq operation
                lf = random.randint(0, n-1)
                rg = random.randint(0, n-1)
                operations.append({'type': 'rmq', 'lf': lf, 'rg': rg})
                
                # Calculate min value
                min_val = float('inf')
                if lf <= rg:
                    for i in range(lf, rg+1):
                        if current_array[i] < min_val:
                            min_val = current_array[i]
                else:
                    for i in range(lf, n):
                        if current_array[i] < min_val:
                            min_val = current_array[i]
                    for i in range(0, rg+1):
                        if current_array[i] < min_val:
                            min_val = current_array[i]
                expected_outputs.append(min_val)
        
        return {
            'n': n,
            'array': array,
            'operations': operations,
            'expected_outputs': expected_outputs
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        array_str = ' '.join(map(str, question_case['array']))
        m = len(question_case['operations'])
        input_lines = [str(n), array_str, str(m)]
        for op in question_case['operations']:
            if op['type'] == 'inc':
                line = f"{op['lf']} {op['rg']} {op['v']}"
            else:
                line = f"{op['lf']} {op['rg']}"
            input_lines.append(line)
        input_data = '\n'.join(input_lines)
        
        prompt = f"""你是一个编程竞赛的参赛者，需要解决以下问题：

给定一个长度为{n}的循环数组，处理一系列操作并输出结果。数组初始值为：{array_str}。

操作类型：
1. inc lf rg v：将循环区间[lf, rg]内的每个元素增加v。当lf <= rg时，区间是连续的；否则包含数组末尾到开头。
2. rmq lf rg：查询循环区间[lf, rg]内的最小值。

输入格式：
{input_data}

请处理所有操作，并按顺序输出每个rmq操作的结果。将结果按顺序放在[answer]和[/answer]标签之间，每个结果占一行。

例如，若结果应为1、0、0，则正确格式为：
[answer]
1
0
0
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = re.compile(r'\[answer\](.*?)\[\/answer\]', re.DOTALL)
        matches = pattern.findall(output)
        if not matches:
            return None
        content = matches[-1].strip()
        solutions = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    solutions.append(int(line))
                except:
                    return None
        return solutions
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_outputs']
