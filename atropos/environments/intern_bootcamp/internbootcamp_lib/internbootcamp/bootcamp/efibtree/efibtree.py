"""# 

### 谜题描述
Let F_k denote the k-th term of Fibonacci sequence, defined as below:

  * F_0 = F_1 = 1
  * for any integer n ≥ 0, F_{n+2} = F_{n+1} + F_n



You are given a tree with n vertices. Recall that a tree is a connected undirected graph without cycles.

We call a tree a Fib-tree, if its number of vertices equals F_k for some k, and at least one of the following conditions holds:

  * The tree consists of only 1 vertex;
  * You can divide it into two Fib-trees by removing some edge of the tree. 



Determine whether the given tree is a Fib-tree or not.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 2 ⋅ 10^5) — the number of vertices in the tree.

Then n-1 lines follow, each of which contains two integers u and v (1≤ u,v ≤ n, u ≠ v), representing an edge between vertices u and v. It's guaranteed that given edges form a tree.

Output

Print \"YES\" if the given tree is a Fib-tree, or \"NO\" otherwise.

You can print your answer in any case. For example, if the answer is \"YES\", then the output \"Yes\" or \"yeS\" will also be considered as correct answer.

Examples

Input


3
1 2
2 3


Output


YES


Input


5
1 2
1 3
1 4
1 5


Output


NO


Input


5
1 3
1 2
4 5
3 4


Output


YES

Note

In the first sample, we can cut the edge (1, 2), and the tree will be split into 2 trees of sizes 1 and 2 correspondently. Any tree of size 2 is a Fib-tree, as it can be split into 2 trees of size 1.

In the second sample, no matter what edge we cut, the tree will be split into 2 trees of sizes 1 and 4. As 4 isn't F_k for any k, it's not Fib-tree.

In the third sample, here is one possible order of cutting the edges so that all the trees in the process are Fib-trees: (1, 3), (1, 2), (4, 5), (3, 4). 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#pragma GCC optimize(\"Ofast\")
#pragma GCC target(\"sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,fma\")
#pragma GCC optimize(\"unroll-loops\")
// #include <atcoder/all>
// #include <bits/stdc++.h>
#include <complex>
#include <queue>
#include <set>
#include <unordered_set>
#include <list>
#include <chrono>
#include <random>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <stack>
#include <iomanip>
#include <fstream>

using namespace std;
// using namespace atcoder;

typedef long long ll;
typedef long double ld;
typedef pair<int, int> p32;
typedef pair<ll, ll> p64;
typedef pair<p64, p64> pp64;
typedef pair<double, double> pdd;
typedef vector<ll> v64;
typedef vector<int> v32;
typedef vector<vector<int>> vv32;
typedef vector<vector<ll>> vv64;
typedef vector<vector<p64>> vvp64;
typedef vector<p64> vp64;
typedef vector<p32> vp32;
ll MOD = 1e9 + 7;
double eps = 1e-12;
#define forn(i, e) for (ll i = 0; i < e; i++)
#define forsn(i, s, e) for (ll i = s; i < e; i++)
#define rforn(i, s) for (ll i = s; i >= 0; i--)
#define rforsn(i, s, e) for (ll i = s; i >= e; i--)
#define ln '\n'
#define dbg(x) cout << #x << \" = \" << x << ln
#define mp make_pair
#define pb push_back
#define fi first
#define se second
#define INF 2e18
#define fast_cin()                    \
    ios_base::sync_with_stdio(false); \
    cin.tie(NULL);                    \
    cout.tie(NULL)
#define all(x) (x).begin(), (x).end()
#define sz(x) ((ll)(x).size())
#define zero ll(0)
#define set_bits(x) __builtin_popcountll(x)
// #define mint modint998244353

ll mpow(ll a, ll b)
{
    if (a == 0)
        return 0;
    if (b == 0)
        return 1;
    ll t1 = mpow(a, b / 2);
    t1 *= t1;
    t1 %= MOD;
    if (b % 2)
        t1 *= a;
    t1 %= MOD;
    return t1;
}

ll mpow(ll a, ll b, ll p)
{
    if (a == 0)
        return 0;
    if (b == 0)
        return 1;
    ll t1 = mpow(a, b / 2, p);
    t1 *= t1;
    t1 %= p;
    if (b % 2)
        t1 *= a;
    t1 %= p;
    return t1;
}

ll modinverse(ll a, ll m)
{
    ll m0 = m;
    ll y = 0, x = 1;
    if (m == 1)
        return 0;
    while (a > 1)
    {
        ll q = a / m;
        ll t = m;
        m = a % m, a = t;
        t = y;
        y = x - q * y;
        x = t;
    }
    if (x < 0)
        x += m0;
    return x;
}

mt19937_64 mt(chrono::steady_clock::now().time_since_epoch().count());

ll range(ll l, ll r)
{
    return l + mt() % (r - l + 1);
}

ll rev(ll v)
{
    return mpow(v, MOD - 2);
}
int LIM = 2e5 + 100;
vv64 adj(LIM);
v64 fib(1000, 1);
v64 S(LIM, 0);
vp64 edges(1);
v64 active(1);
int N;
int node;
void dfs(int n, int pa, int &idx, int k)
{
    S[n] = 1;
    for (int c : adj[n])
    {
        if (active[c] == 0)
            continue;
        p64 temp = edges[c];
        int v;
        if (temp.fi == n)
            v = temp.se;
        else
            v = temp.fi;
        if (v == pa)
            continue;
        dfs(v, n, idx, k);
        if (S[v] == fib[k - 1] || S[v] == fib[k - 2])
        {
            idx = c;
            node = v;
        }
        S[n] += S[v];
    }
}
bool check(int n, int k)
{
    if (k <= 2)
        return true;
    int idx = -1;
    dfs(n, -1, idx, k);
    // cout << \"a\" << endl;
    if (idx == -1)
        return false;
    active[idx] = 0;
    int u = edges[idx].fi, v = edges[idx].se;
    if (u == node)
        swap(u, v);
    // cout << n << \" \" << u << \" \" << v << endl;
    int k1, k2;
    k1 = k - 1;
    k2 = k - 2;
    if (S[v] == fib[k - 1])
        swap(k1, k2);
    bool f1 = check(u, k1), f2 = check(v, k2);
    return f1 && f2;
}
void solve()
{
    ll n;
    cin >> n;
    N = n;
    for (int i = 1; i < n; i++)
    {
        int u, v;
        cin >> u >> v;
        edges.push_back({u, v});
        active.push_back(1);
        adj[u].push_back(i);
        adj[v].push_back(i);
    }
    int k = -1;
    for (int i = 1; i < 1000; i++)
    {
        if (fib[i] == n)
        {
            k = i;
            break;
        }
    }
    if (k == -1)
    {
        cout << \"NO\" << endl;
        return;
    }
    if (check(1, k))
    {
        cout << \"YES\" << endl;
    }
    else
    {
        cout << \"NO\" << endl;
    }
}

int main()
{
    fast_cin();
    ll t = 1;
    for (int i = 2; i < 50; i++)
    {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    // cin >> t;
    forn(i, t)
    {
        // cout << \"Case #\" << i+1 << \": \";
        solve();
    }
    return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Efibtreebootcamp(Basebootcamp):
    def __init__(self, yes_prob=0.5, case_type=None, max_n=200000):
        super().__init__()
        self.yes_prob = yes_prob
        self.case_type = case_type
        self.max_n = max_n
        
        # 预计算斐波那契数列
        self.fib = [1, 1]
        while True:
            next_f = self.fib[-1] + self.fib[-2]
            if next_f > max_n:
                break
            self.fib.append(next_f)
        self.fib_set = set(self.fib)
    
    def _build_fib_tree(self, k):
        """动态构建符合Fib-tree的随机结构"""
        if k == 0 or k == 1:
            return {'nodes': [1], 'edges': []}
        
        # 随机选择分割比例（允许不同分割方向）
        left_k = k-1
        right_k = k-2
        if random.random() < 0.5 and k >= 3:
            left_k, right_k = right_k, left_k
        
        left = self._build_fib_tree(left_k)
        right = self._build_fib_tree(right_k)
        
        # 动态计算节点偏移
        max_left = max(left['nodes'])
        right_nodes = [n + max_left for n in right['nodes']]
        right_edges = [[u+max_left, v+max_left] for u, v in right['edges']]
        
        # 随机选择连接点
        connect_point = random.choice(left['nodes'])
        new_node = max_left + 1 if not right['nodes'] else right_nodes[0]
        
        return {
            'nodes': left['nodes'] + right_nodes,
            'edges': left['edges'] + right_edges + [[connect_point, new_node]]
        }
    
    def case_generator(self):
        # 类型决策逻辑
        if self.case_type is not None:
            generate_yes = (self.case_type == 'YES')
        else:
            generate_yes = random.random() < self.yes_prob
        
        if generate_yes:
            valid_ks = [k for k, f in enumerate(self.fib) if 1 <= f <= self.max_n and k >= 0]
            if not valid_ks:
                valid_ks = [0, 1]
            k = random.choice(valid_ks)
            n = self.fib[k]
            
            # 特殊处理小案例
            if k <= 1:
                return {'n': n, 'edges': [], 'expected': 'YES'}
            
            tree = self._build_fib_tree(k)
            return {
                'n': n,
                'edges': tree['edges'],
                'expected': 'YES'
            }
        else:
            # 生成两种NO案例类型
            if random.random() < 0.5:
                # 类型A：n非斐波那契数
                while True:
                    n = random.randint(1, self.max_n)
                    if n not in self.fib_set:
                        break
                # 生成链式结构
                edges = [[i, i+1] for i in range(1, n)]
                return {'n': n, 'edges': edges, 'expected': 'NO'}
            else:
                # 类型B：斐波那契数但结构非法
                valid_ks = [k for k, f in enumerate(self.fib) if f >=5 and f <= self.max_n]
                k = random.choice(valid_ks)
                n = self.fib[k]
                # 生成星型结构
                edges = [[1, i] for i in range(2, n+1)]
                return {'n': n, 'edges': edges, 'expected': 'NO'}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        edges = question_case['edges']
        edge_list = '\n'.join(f"{u} {v}" for u, v in edges)
        return f"""判断给定的树是否为Fib-tree。规则如下：
1. 顶点数必须是斐波那契数（F_0=1, F_1=1, F_n=F_{{n-1}}+F_{{n-2}}）
2. 单个顶点或可通过移除一条边分割为两个Fib-tree

输入：
{n}
{edge_list}

请将答案（YES/NO）放在[answer]标签内，例如：[answer]YES[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        answer = matches[-1].strip().upper()
        return answer if answer in {'YES', 'NO'} else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        edges = identity['edges']
        
        # 快速判断标记案例
        if 'expected' in identity:
            return solution == identity['expected']
        
        # 构建邻接表
        adj = [[] for _ in range(n+1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # 预计算斐波那契序列
        fib = [1, 1]
        while fib[-1] < n:
            fib.append(fib[-1] + fib[-2])
        if n not in fib:
            return solution == 'NO'
        k = fib.index(n)
        
        # 改进的递归验证算法
        def validate(root, parent):
            size = 1
            valid_splits = []
            
            for child in adj[root]:
                if child == parent:
                    continue
                child_size = validate(child, root)
                if child_size == -1:
                    return -1
                size += child_size
                valid_splits.append(child_size)
            
            # 基准情况
            if size == 1:
                return 1
            
            # 检查当前子树是否可分割
            required = [fib[k-1], fib[k-2]]
            for s in valid_splits:
                if s in required:
                    remaining = size - 1 - s  # 减去当前根节点
                    if remaining in required:
                        return size
            return -1
        
        return (solution == 'YES') == (validate(1, -1) != -1)
