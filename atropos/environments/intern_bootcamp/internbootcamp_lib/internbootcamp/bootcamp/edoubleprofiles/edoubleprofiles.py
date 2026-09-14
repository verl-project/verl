"""# 

### 谜题描述
You have been offered a job in a company developing a large social network. Your first task is connected with searching profiles that most probably belong to the same user.

The social network contains n registered profiles, numbered from 1 to n. Some pairs there are friends (the \"friendship\" relationship is mutual, that is, if i is friends with j, then j is also friends with i). Let's say that profiles i and j (i ≠ j) are doubles, if for any profile k (k ≠ i, k ≠ j) one of the two statements is true: either k is friends with i and j, or k isn't friends with either of them. Also, i and j can be friends or not be friends.

Your task is to count the number of different unordered pairs (i, j), such that the profiles i and j are doubles. Note that the pairs are unordered, that is, pairs (a, b) and (b, a) are considered identical.

Input

The first line contains two space-separated integers n and m (1 ≤ n ≤ 106, 0 ≤ m ≤ 106), — the number of profiles and the number of pairs of friends, correspondingly. 

Next m lines contains descriptions of pairs of friends in the format \"v u\", where v and u (1 ≤ v, u ≤ n, v ≠ u) are numbers of profiles that are friends with each other. It is guaranteed that each unordered pair of friends occurs no more than once and no profile is friends with itself.

Output

Print the single integer — the number of unordered pairs of profiles that are doubles. 

Please do not use the %lld specificator to read or write 64-bit integers in С++. It is preferred to use the %I64d specificator.

Examples

Input

3 3
1 2
2 3
1 3


Output

3


Input

3 0


Output

3


Input

4 1
1 3


Output

2

Note

In the first and second sample any two profiles are doubles.

In the third sample the doubles are pairs of profiles (1, 3) and (2, 4).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 1e6 + 5;
const int MOD = 1e9 + 7;
const int SEED = 29;
long long int p[N];
long long int h[N];
std::vector<pair<int, int> > v;
map<long long int, long long int> mp;
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  int n, m;
  cin >> n >> m;
  p[0] = 1;
  for (int i = 1; i <= n; i++) {
    p[i] = p[i - 1] * SEED;
  }
  for (int i = 0; i < m; i++) {
    int x, y;
    cin >> x >> y;
    v.push_back({x, y});
    h[x] += p[y];
    h[y] += p[x];
  }
  for (int i = 1; i <= n; i++) {
    mp[h[i]]++;
  }
  long long int ans = 0;
  for (auto i : mp) {
    ans += (i.second - 1) * i.second / 2;
  }
  for (auto i : v) {
    if (h[i.first] + p[i.first] == h[i.second] + p[i.second]) ans++;
  }
  cout << ans;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Edoubleprofilesbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=20, seed=None):
        """增强参数校验逻辑"""
        if min_n < 1:
            raise ValueError("min_n must be ≥ 1")
        super().__init__()  # 显式调用父类初始化
        self.min_n = max(min_n, 1)
        self.max_n = max(max_n, self.min_n)
        self.rng = random.Random(seed)

    def case_generator(self):
        """完全重构的用例生成器"""
        n = self.rng.randint(self.min_n, self.max_n)
        
        # 生成所有可能边集
        possible_edges = []
        if n >= 2:
            possible_edges = [(i, j) for i in range(1, n+1) for j in range(i+1, n+1)]
        
        # 安全生成m值
        max_m = len(possible_edges)
        m = self.rng.randint(0, max_m) if max_m > 0 else 0
        
        # 安全采样边集
        edges = []
        if m > 0:
            edges = self.rng.sample(possible_edges, m)
            edges.sort()  # 标准化边存储格式
        
        return {
            'n': n,
            'm': m,
            'edges': edges
        }

    @staticmethod
    def prompt_func(case):
        """增强格式稳定性的提示模板"""
        n, m, edges = case['n'], case['m'], case['edges']
        edge_display = '\n'.join(f"{u} {v}" for u, v in edges) if edges else "无"
        
        return f"""## 双子账号检测任务
某社交网络有{n}个注册账号（编号1-{n}），已知{m}对好友关系：\n{edge_display}

**规则**：
1. 两个不同账号i和j互为双子，当且仅当：
   - 对其他所有账号k，要么k同时是i和j的好友，要么k既不是i也不是j的好友
2. 只需统计无序对(i,j)的数量

请计算满足条件的无序对总数，并将最终数值置于[answer]标签内。示例：
正确答案为5时：[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        """增强数值提取鲁棒性"""
        import re
        candidates = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(candidates[-1]) if candidates else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """带容错的验证逻辑"""
        try:
            return int(solution) == cls._compute_answer(identity)
        except:
            return False

    @classmethod
    def _compute_answer(cls, case):
        """优化哈希算法实现"""
        MOD = 10**18 + 3
        SEED = 29
        n, edges = case['n'], case['edges']
        
        # 初始化哈希基数
        p = [1] * (n + 2)
        for i in range(1, n+1):
            p[i] = (p[i-1] * SEED) % MOD
        
        # 构建特征哈希
        h = defaultdict(int)
        for u, v in edges:
            h[u] = (h[u] + p[v]) % MOD
            h[v] = (h[v] + p[u]) % MOD
        
        # 统计哈希等价类
        counter = defaultdict(int)
        for uid in range(1, n+1):
            counter[h[uid]] += 1
        
        # 计算等价类贡献
        ans = sum(c * (c-1) // 2 for c in counter.values())
        
        # 检查直接边贡献
        processed = set()
        for u, v in edges:
            if (u, v) in processed:
                continue
            if (h[u] + p[u]) % MOD == (h[v] + p[v]) % MOD:
                ans += 1
            processed.add((u, v))
        
        return ans
