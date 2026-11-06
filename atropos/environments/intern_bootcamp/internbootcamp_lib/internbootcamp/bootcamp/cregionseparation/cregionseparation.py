"""# 

### 谜题描述
There are n cities in the Kingdom of Autumn, numbered from 1 to n. People can travel between any two cities using n-1 two-directional roads.

This year, the government decides to separate the kingdom. There will be regions of different levels. The whole kingdom will be the region of level 1. Each region of i-th level should be separated into several (at least two) regions of i+1-th level, unless i-th level is the last level. Each city should belong to exactly one region of each level and for any two cities in the same region, it should be possible to travel between them passing the cities in the same region only.

According to research, for each city i, there is a value a_i, which describes the importance of this city. All regions of the same level should have an equal sum of city importances.

Your task is to find how many plans there are to determine the separation of the regions that all the conditions are satisfied. Two plans are considered different if and only if their numbers of levels are different or there exist two cities in the same region of one level in one plan but in different regions of this level in the other plan. Since the answer may be very large, output it modulo 10^9+7.

Input

The first line contains one integer n (1 ≤ n ≤ 10^6) — the number of the cities.

The second line contains n integers, the i-th of which is a_i (1 ≤ a_i ≤ 10^9) — the value of each city.

The third line contains n-1 integers, p_1, p_2, …, p_{n-1}; p_i (p_i ≤ i) describes a road between cities p_i and i+1.

Output

Print one integer — the number of different plans modulo 10^9+7.

Examples

Input

4
1 1 1 1
1 2 3


Output

4

Input

4
1 1 1 1
1 2 2


Output

2

Input

4
1 2 1 2
1 1 3


Output

3

Note

For the first example, there are 4 different plans:

Plan 1: Level-1: \{1,2,3,4\}.

Plan 2: Level-1: \{1,2,3,4\}, Level-2: \{1,2\},\{3,4\}.

Plan 3: Level-1: \{1,2,3,4\}, Level-2: \{1\},\{2\},\{3\},\{4\}.

Plan 4: Level-1: \{1,2,3,4\}, Level-2: \{1,2\},\{3,4\}, Level-3: \{1\},\{2\},\{3\},\{4\}.

For the second example, there are 2 different plans:

Plan 1: Level-1: \{1,2,3,4\}.

Plan 2: Level-1: \{1,2,3,4\}, Level-2: \{1\},\{2\},\{3\},\{4\}.

For the third example, there are 3 different plans:

Plan 1: Level-1: \{1,2,3,4\}.

Plan 2: Level-1: \{1,2,3,4\}, Level-2: \{1,2\},\{3,4\}.

Plan 3: Level-1: \{1,2,3,4\}, Level-2: \{1,3\},\{2\},\{4\}.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n;
int mod = 1e9 + 7;
int fa[1000100];
long long s[1000100];
long long f[1000100], h[1000100];
long long gcd(long long a, long long b) {
  if (a % b == 0) return b;
  return gcd(b, a % b);
}
int main() {
  scanf(\"%d\", &n);
  for (int i = 1; i <= n; ++i) scanf(\"%lld\", &s[i]);
  for (int i = 1; i < n; ++i) {
    scanf(\"%d\", &fa[i]);
  }
  for (int i = n - 1; i >= 1; --i) {
    s[fa[i]] += s[i + 1];
  }
  long long g;
  for (int i = 1; i <= n; ++i) {
    g = s[1] / gcd(s[1], s[i]);
    if (g <= n) f[g]++;
  }
  for (int i = n / 2; i >= 1; --i) {
    for (int j = i + i; j <= n; j += i) (f[j] += f[i]) %= mod;
  }
  h[1] = 1;
  long long ans = 0;
  for (int i = 1; i <= n; ++i) {
    if (f[i] != i) {
      h[i] = 0;
      continue;
    }
    for (int j = i + i; j <= n; j += i) {
      (h[j] += h[i]) %= mod;
    }
    (ans += h[i]) %= mod;
  }
  printf(\"%lld\n\", ans);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from math import gcd
from bootcamp import Basebootcamp

def compute_answer(n, a, p):
    mod = 10**9 + 7
    if n == 0:
        return 0
    
    s = [0] * (n + 1)  # 1-based indexing
    for i in range(1, n + 1):
        s[i] = a[i - 1]
    
    fa = [0] * (n + 1)
    # 严格保证树结构的有效性
    for i in range(1, n):  # 生成n-1个父节点
        if i-1 < len(p):
            fa[i] = p[i-1]
        else:
            fa[i] = 1  # 默认连接到根节点
    
    # 自底向上聚合子树权重
    for i in range(n-1, 0, -1):
        parent = fa[i]
        s[parent] += s[i+1]
    
    total = s[1]
    if total == 0:
        return 0
    
    # 计算因数分布
    f = [0]*(n+2)
    for i in range(1, n+1):
        current_gcd = gcd(total, s[i])
        g = total // current_gcd
        if g <= n:
            f[g] += 1
    
    # 倍数传播
    for i in range(n//2, 0, -1):
        j = 2*i
        while j <= n:
            f[j] = (f[j] + f[i]) % mod
            j += i
    
    # DP计算方案数 (修复最终答案计算)
    h = [0]*(n+2)
    h[1] = 1
    ans = 0
    for i in range(1, n+1):
        if f[i] != i:
            h[i] = 0
            continue
        j = 2*i
        while j <= n:
            h[j] = (h[j] + h[i]) % mod
            j += i
        ans = (ans + h[i]) % mod
    
    return ans

class Cregionseparationbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_n = params.get('max_n', 8)
        self.value_range = params.get('value_range', (1, 5))
        self.force_tree_shape = params.get('force_tree_shape', True)
    
    def case_generator(self):
        while True:  # 确保生成有效树结构
            n = random.randint(4, self.max_n)
            
            # 生成重要性值数组（确保非零）
            a = [random.randint(*self.value_range) for _ in range(n)]
            
            # 生成严格合法的树结构
            p = []
            nodes = [1]  # 已存在的节点
            for i in range(1, n):  # 生成城市i+1的父节点
                p_i = random.choice(nodes[:i])  # 保证p_i <= i
                p.append(p_i)
                nodes.append(i+1)
            
            # 计算标准答案
            try:
                answer = compute_answer(n, a, p)
                return {
                    'n': n,
                    'a': a,
                    'p': p,
                    'answer': answer
                }
            except:
                continue
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        p = question_case['p']
        input_str = f"{n}\n{' '.join(map(str, a))}\n{' '.join(map(str, p))}"
        prompt = f"""You are in a programming competition. Solve this problem and wrap your answer with [answer][/answer].

Problem:
Kingdom of Autumn has {n} cities connected as a tree. Each city has importance a_i. We need to partition the kingdom into hierarchical regions where each level's regions have equal importance sums. Count valid partition schemes modulo 1e9+7.

Input:
{input_str}

Output:
A single integer - the answer modulo 1e9+7. Format: [answer]number[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
