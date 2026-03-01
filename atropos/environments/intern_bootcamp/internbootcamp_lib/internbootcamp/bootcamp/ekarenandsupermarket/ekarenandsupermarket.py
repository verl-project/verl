"""# 

### 谜题描述
On the way home, Karen decided to stop by the supermarket to buy some groceries.

<image>

She needs to buy a lot of goods, but since she is a student her budget is still quite limited. In fact, she can only spend up to b dollars.

The supermarket sells n goods. The i-th good can be bought for ci dollars. Of course, each good can only be bought once.

Lately, the supermarket has been trying to increase its business. Karen, being a loyal customer, was given n coupons. If Karen purchases the i-th good, she can use the i-th coupon to decrease its price by di. Of course, a coupon cannot be used without buying the corresponding good.

There is, however, a constraint with the coupons. For all i ≥ 2, in order to use the i-th coupon, Karen must also use the xi-th coupon (which may mean using even more coupons to satisfy the requirement for that coupon).

Karen wants to know the following. What is the maximum number of goods she can buy, without exceeding her budget b?

Input

The first line of input contains two integers n and b (1 ≤ n ≤ 5000, 1 ≤ b ≤ 109), the number of goods in the store and the amount of money Karen has, respectively.

The next n lines describe the items. Specifically:

  * The i-th line among these starts with two integers, ci and di (1 ≤ di < ci ≤ 109), the price of the i-th good and the discount when using the coupon for the i-th good, respectively. 
  * If i ≥ 2, this is followed by another integer, xi (1 ≤ xi < i), denoting that the xi-th coupon must also be used before this coupon can be used. 

Output

Output a single integer on a line by itself, the number of different goods Karen can buy, without exceeding her budget.

Examples

Input

6 16
10 9
10 5 1
12 2 1
20 18 3
10 2 3
2 1 5


Output

4


Input

5 10
3 1
3 1 1
3 1 2
3 1 3
3 1 4


Output

5

Note

In the first test case, Karen can purchase the following 4 items:

  * Use the first coupon to buy the first item for 10 - 9 = 1 dollar. 
  * Use the third coupon to buy the third item for 12 - 2 = 10 dollars. 
  * Use the fourth coupon to buy the fourth item for 20 - 18 = 2 dollars. 
  * Buy the sixth item for 2 dollars. 



The total cost of these goods is 15, which falls within her budget. Note, for example, that she cannot use the coupon on the sixth item, because then she should have also used the fifth coupon to buy the fifth item, which she did not do here.

In the second test case, Karen has enough money to use all the coupons and purchase everything.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
void fre() {
  freopen(\"c://test//input.in\", \"r\", stdin);
  freopen(\"c://test//output.out\", \"w\", stdout);
}
template <class T1, class T2>
inline void gmax(T1 &a, T2 b) {
  if (b > a) a = b;
}
template <class T1, class T2>
inline void gmin(T1 &a, T2 b) {
  if (b < a) a = b;
}
const int N = 5005, M = 0, Z = 1e9 + 7;
const int inf = 0x3f3f3f3f;
template <class T1, class T2>
inline void gadd(T1 &a, T2 b) {
  a = (a + b) % Z;
}
int casenum, casei;
int n, m;
int f[N][N];
int d[N][N];
int v[N];
int c[N];
vector<int> a[N];
int sz[N];
void dfs(int x) {
  f[x][0] = d[x][0] = sz[x] = 0;
  for (auto y : a[x]) {
    dfs(y);
    for (int i = sz[x]; ~i; --i) {
      for (int j = 1; j <= sz[y]; ++j) {
        gmin(f[x][i + j], f[x][i] + f[y][j]);
        gmin(f[x][i + j], f[x][i] + d[y][j]);
        gmin(d[x][i + j], d[x][i] + d[y][j]);
      }
    }
    sz[x] += sz[y];
  }
  ++sz[x];
  for (int i = sz[x]; i; --i) {
    f[x][i] = min(f[x][i - 1] + v[x] - c[x], inf);
    gmin(d[x][i], d[x][i - 1] + v[x]);
  }
}
int main() {
  while (~scanf(\"%d%d\", &n, &m)) {
    memset(f, 63, sizeof(f));
    memset(d, 63, sizeof(d));
    for (int i = 1; i <= n; ++i) {
      a[i].clear();
      scanf(\"%d%d\", &v[i], &c[i]);
      if (i > 1) {
        int fa;
        scanf(\"%d\", &fa);
        a[fa].push_back(i);
      }
    }
    dfs(1);
    int ans = 0;
    for (int i = 1; i <= n; ++i) {
      if (f[1][i] <= m || d[1][i] <= m) {
        gmax(ans, i);
      }
    }
    printf(\"%d\n\", ans);
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

class Ekarenandsupermarketbootcamp(Basebootcamp):
    def __init__(self, max_n=20, c_range=(5, 20), d_ratio=0.5, **kwargs):
        self.max_n = max_n
        self.c_range = c_range
        self.d_ratio = d_ratio
        super().__init__(**kwargs)
    
    def case_generator(self):
        """Generate a valid test case with tree-structured coupon dependencies"""
        n = random.randint(1, self.max_n)
        items = []
        dependency_tree = {1: []}
        
        for i in range(1, n+1):
            ci = random.randint(*self.c_range)
            di = random.randint(max(1, int(ci * self.d_ratio)), ci-1)
            
            if i == 1:
                items.append({'c': ci, 'd': di})
            else:
                xi = random.choice(list(dependency_tree.keys()))
                items.append({'c': ci, 'd': di, 'x': xi})
                dependency_tree[i] = []
                dependency_tree[xi].append(i)
        
        # Calculate reasonable budget range
        min_cost = sum(item['c'] - item['d'] for item in items)
        max_cost = sum(item['c'] for item in items)
        b = random.randint(
            int(min_cost * 0.5),
            max_cost + random.randint(0, sum(item['d'] for item in items))
        )
        
        return {
            'n': n,
            'b': b,
            'items': items,
            'dependency_tree': dependency_tree
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [f"{question_case['n']} {question_case['b']}"]
        for i, item in enumerate(question_case['items'], 1):
            if i == 1:
                input_lines.append(f"{item['c']} {item['d']}")
            else:
                input_lines.append(f"{item['c']} {item['d']} {item['x']}")
        
        return f"""Karen wants to maximize purchased goods with coupons under dependency constraints. Rules:
1. Each product has a coupon that reduces price by di (must buy the product to use)
2. For i≥2, using coupon i requires using coupon xi (which may have its own dependencies)
3. Budget cannot exceed b dollars
4. Each product can be bought at most once

Input format:
n b
c1 d1 (first product, no dependency)
c2 d2 x2 (subsequent products show dependency)
...
cn dn xn

Current input:
{chr(10).join(input_lines)}

Calculate the maximum number of items Karen can buy within budget. Put only the integer answer within [answer]...[/answer] tags."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        class Solver:
            def __init__(self, n, b, items):
                self.n = n
                self.b = b
                self.children = [[] for _ in range(n+2)]
                self.v = [0]*(n+2)
                self.c = [0]*(n+2)
                
                for i in range(1, n+1):
                    item = items[i-1]
                    self.v[i] = item['c']
                    self.c[i] = item['d']
                    if i > 1:
                        self.children[item['x']].append(i)
                
                self.INF = float('inf')

            def solve(self):
                dp = [{} for _ in range(self.n+2)]
                
                def dfs(u):
                    node = {'use': {}, 'nouse': {}}
                    node['use'][0] = 0
                    node['nouse'][0] = 0
                    
                    for v in self.children[u]:
                        child = dfs(v)
                        merged_use = {}
                        merged_nouse = {}
                        
                        # Merge child states into current node
                        for ku in node['use']:
                            for kv in child['use']:
                                key = ku + kv
                                val = node['use'][ku] + child['use'][kv]
                                if val <= self.b:
                                    merged_use[key] = min(merged_use.get(key, self.INF), val)
                                
                            for kv in child['nouse']:
                                key = ku + kv
                                val = node['use'][ku] + child['nouse'][kv]
                                if val <= self.b:
                                    merged_use[key] = min(merged_use.get(key, self.INF), val)
                        
                        for ku in node['nouse']:
                            for kv in child['nouse']:
                                key = ku + kv
                                val = node['nouse'][ku] + child['nouse'][kv]
                                if val <= self.b:
                                    merged_nouse[key] = min(merged_nouse.get(key, self.INF), val)
                        
                        # Update current node's states
                        new_use = {}
                        for k in merged_use:
                            new_use[k] = min(node['use'].get(k, self.INF), merged_use[k])
                        for k in node['use']:
                            new_use[k] = min(new_use.get(k, self.INF), node['use'][k])
                        
                        new_nouse = {}
                        for k in merged_nouse:
                            new_nouse[k] = min(node['nouse'].get(k, self.INF), merged_nouse[k])
                        for k in node['nouse']:
                            new_nouse[k] = min(new_nouse.get(k, self.INF), node['nouse'][k])
                        
                        node['use'] = new_use
                        node['nouse'] = new_nouse
                    
                    # Add current node's options
                    new_use = {}
                    for k in node['use']:
                        newk = k + 1
                        cost = node['use'][k] + (self.v[u] - self.c[u])
                        if cost <= self.b:
                            new_use[newk] = min(node['use'].get(newk, self.INF), cost)
                    node['use'] = {**node['use'], **new_use}
                    
                    new_nouse = {}
                    for k in node['nouse']:
                        newk = k + 1
                        cost = node['nouse'][k] + self.v[u]
                        if cost <= self.b:
                            new_nouse[newk] = min(node['nouse'].get(newk, self.INF), cost)
                    node['nouse'] = {**node['nouse'], **new_nouse}
                    
                    return node
                
                root = dfs(1)
                max_items = 0
                for k in root['use']:
                    if k > max_items and root['use'][k] <= self.b:
                        max_items = k
                for k in root['nouse']:
                    if k > max_items and root['nouse'][k] <= self.b:
                        max_items = k
                return max_items

        n = identity['n']
        b = identity['b']
        items = identity['items']
        return solution == Solver(n, b, items).solve()
