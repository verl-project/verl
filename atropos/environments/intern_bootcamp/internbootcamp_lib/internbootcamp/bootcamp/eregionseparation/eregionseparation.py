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
inline int read() {
  int x = 0, fu = 0;
  char ch = getchar();
  for (; ch < 48 || ch > 57; ch = getchar()) fu |= (ch == '-');
  for (; ch > 47 && ch < 58; ch = getchar()) x = x * 10 + ch - '0';
  return fu ? -x : x;
}
inline void add(int &x, const int &y) {
  x += y;
  x = x < 1000000007 ? x : x - 1000000007;
}
inline long long gcd(long long a, long long b) {
  return b == 0 ? a : gcd(b, a % b);
}
int n, fa[1000005], f[1000005], dp[1000005], ans;
long long a[1000005];
int main() {
  n = read(), dp[1] = 1;
  for (int i = 1; i <= n; i++) a[i] = read();
  for (int i = 2; i <= n; i++) fa[i] = read();
  for (int i = n; i >= 2; i--) a[fa[i]] += a[i];
  for (int i = n; i >= 1; i--) a[i] = a[1] / gcd(a[1], a[i]);
  for (int i = 1; i <= n; i++) a[i] > n ? 0 : f[a[i]]++;
  for (int i = n; i >= 1; i--)
    for (int j = i << 1; j <= n; j += i) add(f[j], f[i]);
  for (int i = 1; i <= n; i++)
    if (f[i] == i) {
      add(ans, dp[i]);
      for (int j = i << 1; j <= n; j += i) add(dp[j], dp[i]);
    }
  printf(\"%d\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Eregionseparationbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=6, **params):
        super().__init__(**params)
        self.min_n = min_n  # 最小城市数量
        self.max_n = max_n  # 最大城市数量

    def case_generator(self):
        """生成随机有效的谜题实例"""
        n = random.randint(self.min_n, self.max_n)
        a = [1] * n  # 使用全1数组简化计算
        
        # 生成合法树结构
        p = []
        for city in range(2, n+1):
            p.append(random.randint(1, city-1))  # 父节点必须 <= city-1
        
        return {
            'n': n,
            'a': a,
            'p': p,
            'expected_answer': self._compute_answer(n, a, p)
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        p = question_case['p']
        return f"""你是秋之王国的规划师，需要制定区域划分方案。王国包含{n}个城市（编号1-{n}），城市由以下道路连接（每条p_i表示i+1号城市与p_i号城市相连）：{p}。
各城市重要值依次为：{a}。

规则要求：
1. 层级划分必须满足每层区域可继续分割（最后层级除外）
2. 同层级所有区域重要值之和必须相等
3. 区域必须保持连通性

请计算符合要求的划分方案总数（模10^9+7），将最终答案放在[answer][/answer]标签内。例如：
[answer] 42 [/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['expected_answer']
        except:
            return False

    @staticmethod
    def _compute_answer(n, a_list, p_list):
        """动态计算正确答案"""
        if n == 0:
            return 0
        
        # 初始化数据结构
        a = [0] * (n + 1)
        for i in range(n):
            a[i+1] = a_list[i]
        
        # 构建父节点数组
        fa = [0] * (n + 1)
        for idx, parent in enumerate(p_list):
            fa[idx+2] = parent  # p_list对应城市2到n的父节点
        
        # 自底向上计算子树和
        for i in range(n, 1, -1):
            a[fa[i]] += a[i]
        
        S = a[1]
        if S == 0:
            return 0
        
        # 处理a数组
        for i in range(1, n+1):
            a[i] = S // math.gcd(S, a[i])
        
        # 计算频率数组
        freq = [0] * (n + 2)
        for i in range(1, n+1):
            if a[i] <= n:
                freq[a[i]] += 1
        
        # 因数叠加
        for i in range(n, 0, -1):
            j = 2 * i
            while j <= n:
                freq[j] += freq[i]
                j += i
        
        # 动态规划求解
        dp = [0] * (n + 2)
        dp[1] = 1
        ans = 0
        for i in range(1, n+1):
            if freq[i] == i:
                ans = (ans + dp[i]) % MOD
                j = 2 * i
                while j <= n:
                    dp[j] = (dp[j] + dp[i]) % MOD
                    j += i
        
        return ans % MOD
