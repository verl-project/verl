"""# 

### 谜题描述
In the Isle of Guernsey there are n different types of coins. For each i (1 ≤ i ≤ n), coin of type i is worth ai cents. It is possible that ai = aj for some i and j (i ≠ j). 

Bessie has some set of these coins totaling t cents. She tells Jessie q pairs of integers. For each i (1 ≤ i ≤ q), the pair bi, ci tells Jessie that Bessie has a strictly greater number of coins of type bi than coins of type ci. It is known that all bi are distinct and all ci are distinct. 

Help Jessie find the number of possible combinations of coins Bessie could have. Two combinations are considered different if there is some i (1 ≤ i ≤ n), such that the number of coins Bessie has of type i is different in the two combinations. Since the answer can be very large, output it modulo 1000000007 (109 + 7). 

If there are no possible combinations of coins totaling t cents that satisfy Bessie's conditions, output 0.

Input

The first line contains three space-separated integers, n, q and t (1 ≤ n ≤ 300; 0 ≤ q ≤ n; 1 ≤ t ≤ 105). The second line contains n space separated integers, a1, a2, ..., an (1 ≤ ai ≤ 105). The next q lines each contain two distinct space-separated integers, bi and ci (1 ≤ bi, ci ≤ n; bi ≠ ci).

It's guaranteed that all bi are distinct and all ci are distinct.

Output

A single integer, the number of valid coin combinations that Bessie could have, modulo 1000000007 (109 + 7).

Examples

Input

4 2 17
3 1 2 5
4 2
3 4


Output

3


Input

3 2 6
3 1 1
1 2
2 3


Output

0


Input

3 2 10
1 2 3
1 2
2 1


Output

0

Note

For the first sample, the following 3 combinations give a total of 17 cents and satisfy the given conditions: {0 of type 1, 1 of type 2, 3 of type 3, 2 of type 4}, {0, 0, 6, 1}, {2, 0, 3, 1}.

No other combinations exist. Note that even though 4 occurs in both bi and ci,  the problem conditions are still satisfied because all bi are distinct and all ci are distinct.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
namespace FGF {
const int N = 5e5 + 5, mo = 1e9 + 7;
long long n, a[N], m, q, fa[N], vis[N], dep[N], ins[N], dp[N], sum[N];
vector<int> g[N];
void dfs(int u) {
  ins[u] = 1;
  for (auto v : g[u]) sum[v] = a[v] + sum[u], dfs(v), dep[u] = dep[v] + 1;
  m -= dep[u] * a[u];
}
void work() {
  scanf(\"%lld%lld%lld\", &n, &q, &m);
  for (int i = 1; i <= n; i++) scanf(\"%lld\", &a[i]);
  for (int i = 1, u, v; i <= q; i++)
    scanf(\"%d%d\", &u, &v), g[u].push_back(v), vis[v] = 1;
  for (int i = 1; i <= n; i++)
    if (!vis[i]) sum[i] = a[i], dfs(i);
  bool fl = 1;
  for (int i = 1; i <= n; i++) fl &= ins[i];
  if (m < 0 || !fl) {
    puts(\"0\");
    return;
  }
  dp[0] = 1;
  for (int i = 1; i <= n; i++)
    for (int j = sum[i]; j <= m; j++) (dp[j] += dp[j - sum[i]]) %= mo;
  printf(\"%lld\", dp[m]);
}
}  // namespace FGF
int main() {
  FGF::work();
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
import re

MOD = 10**9 + 7

def solve(n, q, t, a_list, constraints):
    """完整实现的解题算法"""
    # 初始化图结构
    g = [[] for _ in range(n+1)]
    in_degree = [0]*(n+1)
    for u, v in constraints:
        g[u].append(v)
        in_degree[v] += 1

    # 拓扑排序检测环
    queue = deque()
    topo_order = []
    for u in range(1, n+1):
        if in_degree[u] == 0:
            queue.append(u)
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in g[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    if len(topo_order) != n:
        return 0  # 存在环

    # 计算依赖关系和最小金额
    dep = [0]*(n+1)
    sum_ = [0]*(n+1)
    for u in reversed(topo_order):
        sum_[u] = a_list[u-1]
        max_child_dep = 0
        for v in g[u]:
            sum_[u] += sum_[v]
            if dep[v] > max_child_dep:
                max_child_dep = dep[v]
        dep[u] = max_child_dep + 1

    min_t = sum(a_list[u-1] * dep[u] for u in topo_order)
    if t < min_t:
        return 0

    # 动态规划计算组合数
    target = t - min_t
    dp = [0]*(target+1)
    dp[0] = 1
    for u in topo_order:
        s = sum_[u]
        for j in range(s, target+1):
            dp[j] = (dp[j] + dp[j - s]) % MOD
    
    return dp[target] % MOD if target >=0 else 0

class Ccointroublesbootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_q=3, max_a=5, max_t=1000):
        self.max_n = max_n
        self.max_q = max_q
        self.max_a = max_a
        self.max_t = max_t

    def case_generator(self):
        """改进的测试用例生成器"""
        for _ in range(100):  # 最多尝试次数
            # 生成有效约束条件
            n = random.randint(1, self.max_n)
            a = [random.randint(1, self.max_a) for _ in range(n)]
            
            # 生成拓扑约束
            nodes = list(range(1, n+1))
            random.shuffle(nodes)
            constraints = []
            used_bi = set()
            used_ci = set()
            
            # 保证bi/ci唯一的有效生成方式
            available_bi = nodes.copy()
            available_ci = nodes.copy()
            for _ in range(min(self.max_q, n//2)):
                if not available_bi or not available_ci:
                    break
                bi = random.choice(available_bi)
                available_bi.remove(bi)
                ci_candidates = [c for c in available_ci if c != bi and c not in used_ci]
                if not ci_candidates:
                    continue
                ci = random.choice(ci_candidates)
                available_ci.remove(ci)
                constraints.append((bi, ci))
                used_bi.add(bi)
                used_ci.add(ci)
            
            q = len(constraints)
            
            # 计算最小金额
            min_t = self._calculate_min_t(n, a, constraints)
            if min_t is None or min_t > self.max_t:
                continue
            
            # 生成有效金额
            max_add = self.max_t - min_t
            t = min_t + random.randint(0, max(0, max_add))
            
            # 验证案例有效性
            case = {
                'n': n,
                'q': q,
                't': t,
                'a': a.copy(),
                'constraints': constraints.copy()
            }
            
            # 计算标准答案
            try:
                ans = solve(n, q, t, a, constraints)
                if ans >= 0:
                    case['correct_answer'] = ans
                    return case
            except:
                continue
        return None  # 极端情况下返回空

    def _calculate_min_t(self, n, a, constraints):
        """辅助函数：计算最小金额"""
        try:
            temp_g = [[] for _ in range(n+1)]
            for u, v in constraints:
                temp_g[u].append(v)
            
            # 计算拓扑深度
            depth = [0]*(n+1)
            for u in range(n, 0, -1):
                max_child = 0
                for v in temp_g[u]:
                    max_child = max(max_child, depth[v])
                depth[u] = max_child + 1
            
            return sum(a[u-1] * depth[u] for u in range(1, n+1))
        except:
            return None

    @staticmethod
    def prompt_func(case):
        """优化的问题描述生成"""
        constraints = "\n".join([f"- Type {b} coins > Type {c} coins" 
                               for b, c in case['constraints']])
        return f"""Calculate valid coin combinations with:
- {case['n']} coin types: {', '.join(map(str, case['a']))}
- Total required: {case['t']} cents
- Constraints:
{constraints}

Output the answer modulo 1e9+7 within [answer]...[/answer]"""

    @staticmethod
    def extract_output(output):
        """加强的答案提取"""
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) % MOD if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """改进的验证逻辑"""
        try:
            expected = identity['correct_answer']
            return (int(solution) % MOD) == (expected % MOD)
        except:
            return False
