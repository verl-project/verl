"""# 

### 谜题描述
Drazil likes heap very much. So he created a problem with heap:

There is a max heap with a height h implemented on the array. The details of this heap are the following:

This heap contains exactly 2^h - 1 distinct positive non-zero integers. All integers are distinct. These numbers are stored in the array a indexed from 1 to 2^h-1. For any 1 < i < 2^h, a[i] < a[\left ⌊{i/2}\right ⌋].

Now we want to reduce the height of this heap such that the height becomes g with exactly 2^g-1 numbers in heap. To reduce the height, we should perform the following action 2^h-2^g times:

Choose an index i, which contains an element and call the following function f in index i:

<image>

Note that we suppose that if a[i]=0, then index i don't contain an element.

After all operations, the remaining 2^g-1 element must be located in indices from 1 to 2^g-1. Now Drazil wonders what's the minimum possible sum of the remaining 2^g-1 elements. Please find this sum and find a sequence of the function calls to achieve this value.

Input

The first line of the input contains an integer t (1 ≤ t ≤ 70 000): the number of test cases.

Each test case contain two lines. The first line contains two integers h and g (1 ≤ g < h ≤ 20). The second line contains n = 2^h-1 distinct positive integers a[1], a[2], …, a[n] (1 ≤ a[i] < 2^{20}). For all i from 2 to 2^h - 1, a[i] < a[\left ⌊{i/2}\right ⌋].

The total sum of n is less than 2^{20}.

Output

For each test case, print two lines.

The first line should contain one integer denoting the minimum sum after reducing the height of heap to g. The second line should contain 2^h - 2^g integers v_1, v_2, …, v_{2^h-2^g}. In i-th operation f(v_i) should be called.

Example

Input


2
3 2
7 6 3 5 4 2 1
3 2
7 6 5 4 3 2 1


Output


10
3 2 3 1
8
2 1 3 1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
#pragma GCC optimize(\"O3\")
#pragma GCC optimize(\"unroll-loops\")
#pragma comment(linker, \"/STACK:2000000\")
using namespace std;
using ll = long long;
using db = long double;
using ii = pair<int, int>;
const int N = 3e5 + 5, LG = 19, MOD = 998244353;
const int SQ = 320;
const long double EPS = 1e-7;
int d, m, t;
bool deletable[1 << 22];
int h, g, a[1 << 22];
int pos[1 << 22];
void dfs(int node, int height) {
  if (height > g) return;
  deletable[node] = false;
  dfs(node << 1, height + 1);
  dfs(node << 1 | 1, height + 1);
}
int get(int x) {
  if (a[x << 1] == 0 && a[x << 1 | 1] == 0)
    return x;
  else {
    if (a[x << 1] > a[x << 1 | 1])
      return get(x << 1);
    else
      return get(x << 1 | 1);
  }
  return -1;
}
void apply(int x) {
  if (a[x << 1] == 0 && a[x << 1 | 1] == 0) {
    a[x] = 0;
  } else {
    a[x] = max(a[x << 1], a[x << 1 | 1]);
    pos[a[x]] = x;
    if (a[x << 1] > a[x << 1 | 1])
      return apply(x << 1);
    else
      return apply(x << 1 | 1);
  }
}
vector<int> vt;
int ptr;
vector<int> ans;
bool kill() {
  assert(ptr < vt.size());
  int x = vt[ptr++];
  int y = get(pos[x]);
  if (y == -1) {
    cout << -1 << '\n';
    exit(0);
  }
  if (deletable[y]) {
    ans.push_back(pos[x]);
    apply(pos[x]);
    return true;
  } else
    return false;
}
int32_t main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cin >> t;
  while (t--) {
    cin >> h >> g;
    for (int i = 1; i < 1 << h; i++) {
      cin >> a[i];
      a[i + i] = a[i + i + 1] = 0;
      pos[a[i]] = i;
      vt.push_back(a[i]);
    }
    sort(vt.rbegin(), vt.rend());
    fill(deletable, deletable + (1 << h), 1);
    dfs(1, 1);
    int cur = (1 << h);
    while (cur != (1 << g)) {
      cur -= kill();
    }
    cout << accumulate(a, a + (1 << h), 0ll) << '\n';
    for (auto x : ans) cout << x << ' ';
    cout << '\n';
    ans.clear();
    vt.clear();
    ptr = 0;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import heapq
import re
from itertools import count
from bootcamp import Basebootcamp

class Cdrazillikesheapbootcamp(Basebootcamp):
    def __init__(self, h_range=(3, 5)):
        self.min_h, self.max_h = h_range  # 控制合理的堆高度范围

    def case_generator(self):
        h = random.randint(self.min_h, self.max_h)
        g = random.randint(max(1, h-2), h-1)  # 保证足够的操作次数
        n = (1 << h) - 1
        
        # 生成满足最大堆条件的数组
        values = random.sample(range(1, (1 << 20)), n)
        heapq._heapify_max(values)
        
        # 构建完全二叉树结构
        heap = [0]*n
        for i in range(n):
            heap[i] = values[i]
            if i > 0:
                parent = (i-1) // 2
                assert heap[parent] > heap[i], "Invalid heap structure"
        
        return {
            'h': h,
            'g': g,
            'array': heap
        }

    @staticmethod
    def prompt_func(case) -> str:
        prompt = (
            f"## 堆高度缩减问题\n"
            f"初始堆高度为{case['h']}，需要缩减至高度{case['g']}\n"
            f"堆数组：{' '.join(map(str, case['array']))}\n\n"
            "**操作规则**：\n"
            "1. 每次操作选择节点i调用f(i)\n"
            "2. f(i)将节点值替换为较大子节点的值，递归处理直到叶子节点置零\n"
            "3. 最终保留的节点必须构成高度{case['g']}的有效最大堆\n\n"
            "**答案格式**：\n"
            "[answer]\n"
            "总和值\n"
            "操作序列（空格分隔）\n"
            "[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(text):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
            
        last_match = matches[-1].strip()
        try:
            sum_line, ops_line = last_match.split('\n')[:2]
            return {
                'sum': int(sum_line.strip()),
                'operations': list(map(int, ops_line.strip().split()))
            }
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, case):
        if not solution or 'sum' not in solution or 'operations' not in solution:
            return False

        h, g = case['h'], case['g']
        expected_ops = (1 << h) - (1 << g)
        if len(solution['operations']) != expected_ops:
            return False

        # 构建堆结构
        size = (1 << h)
        a = [0] * size
        for i, val in enumerate(case['array'], 1):
            a[i] = val

        # 模拟操作
        for op in solution['operations']:
            if op >= size or a[op] == 0:
                return False

            current = op
            while True:
                left = 2 * current
                right = 2 * current + 1
                
                # 边界检查
                if left >= size:
                    a[current] = 0
                    break
                
                # 查找最大子节点
                max_child = left if a[left] > a[right] else right
                if a[max_child] == 0:
                    a[current] = 0
                    break
                
                # 替换节点值
                a[current] = a[max_child]
                current = max_child

        # 验证最终堆结构
        valid_size = (1 << g) - 1
        for i in range(1, valid_size+1):
            if a[i] == 0:
                return False

        for i in range(1, valid_size+1):
            left = 2*i
            right = 2*i+1
            if left <= valid_size and a[i] < a[left]:
                return False
            if right <= valid_size and a[i] < a[right]:
                return False

        return sum(a[1:valid_size+1]) == solution['sum']
