"""# 

### 谜题描述
Welcome! Everything is fine.

You have arrived in The Medium Place, the place between The Good Place and The Bad Place. You are assigned a task that will either make people happier or torture them for eternity.

You have a list of k pairs of people who have arrived in a new inhabited neighborhood. You need to assign each of the 2k people into one of the 2k houses. Each person will be the resident of exactly one house, and each house will have exactly one resident.

Of course, in the neighborhood, it is possible to visit friends. There are 2k - 1 roads, each of which connects two houses. It takes some time to traverse a road. We will specify the amount of time it takes in the input. The neighborhood is designed in such a way that from anyone's house, there is exactly one sequence of distinct roads you can take to any other house. In other words, the graph with the houses as vertices and the roads as edges is a tree.

The truth is, these k pairs of people are actually soulmates. We index them from 1 to k. We denote by f(i) the amount of time it takes for the i-th pair of soulmates to go to each other's houses.

As we have said before, you will need to assign each of the 2k people into one of the 2k houses. You have two missions, one from the entities in The Good Place and one from the entities of The Bad Place. Here they are:

  * The first mission, from The Good Place, is to assign the people into the houses such that the sum of f(i) over all pairs i is minimized. Let's define this minimized sum as G. This makes sure that soulmates can easily and efficiently visit each other; 
  * The second mission, from The Bad Place, is to assign the people into the houses such that the sum of f(i) over all pairs i is maximized. Let's define this maximized sum as B. This makes sure that soulmates will have a difficult time to visit each other. 



What are the values of G and B?

Input

The first line of input contains a single integer t (1 ≤ t ≤ 500) denoting the number of test cases. The next lines contain descriptions of the test cases.

The first line of each test case contains a single integer k denoting the number of pairs of people (1 ≤ k ≤ 10^5). The next 2k - 1 lines describe the roads; the i-th of them contains three space-separated integers a_i, b_i, t_i which means that the i-th road connects the a_i-th and b_i-th houses with a road that takes t_i units of time to traverse (1 ≤ a_i, b_i ≤ 2k, a_i ≠ b_i, 1 ≤ t_i ≤ 10^6). It is guaranteed that the given roads define a tree structure.

It is guaranteed that the sum of the k in a single file is at most 3 ⋅ 10^5.

Output

For each test case, output a single line containing two space-separated integers G and B. 

Example

Input


2
3
1 2 3
3 2 4
2 4 3
4 5 6
5 6 5
2
1 2 1
1 3 2
1 4 3


Output


15 33
6 6

Note

For the sample test case, we have a minimum sum equal to G = 15. One way this can be achieved is with the following assignment:

  * The first pair of people get assigned to houses 5 and 6, giving us f(1) = 5; 
  * The second pair of people get assigned to houses 1 and 4, giving us f(2) = 6; 
  * The third pair of people get assigned to houses 3 and 2, giving us f(3) = 4. 



Note that the sum of the f(i) is 5 + 6 + 4 = 15. 

We also have a maximum sum equal to B = 33. One way this can be achieved is with the following assignment:

  * The first pair of people get assigned to houses 1 and 4, giving us f(1) = 6; 
  * The second pair of people get assigned to houses 6 and 2, giving us f(2) = 14; 
  * The third pair of people get assigned to houses 3 and 5, giving us f(3) = 13. 



Note that the sum of the f(i) is 6 + 14 + 13 = 33. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long gcd(long long a, long long b) {
  if (a > b) swap(a, b);
  if (a == 0) return b;
  return gcd(b % a, a);
}
long long powerMod(long long x, long long y) {
  long long res = 1;
  x %= 1000000007;
  while (y > 0) {
    if (y & 1) res = (res * x) % 1000000007;
    y = y >> 1;
    x = (x * x) % 1000000007;
  }
  return res % 1000000007;
}
string binary(long long s) {
  string res = \"\";
  while (s != 0) {
    res += (char)('0' + s % 2);
    s /= 2;
  }
  reverse(res.begin(), res.end());
  return res;
}
vector<vector<pair<long long, long long> > > adj;
vector<bool> vis;
vector<long long> cc;
vector<pair<long long, long long> > edges;
map<pair<long long, long long>, long long> mp;
long long n;
long long dfs(int y) {
  vis[y] = true;
  long long k = 1LL;
  for (pair<long long, long long> p : adj[y]) {
    if (!vis[p.first]) {
      long long x = dfs(p.first);
      cc[p.second] = min(n - x, x);
      k += x;
    }
  }
  return k;
}
int main() {
  ios::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  ;
  long long t, m, i, j, k, l, p, mx, mn;
  cin >> t;
  while (t--) {
    cin >> n;
    n *= 2LL;
    adj.resize(n);
    vis.resize(n, false);
    cc.resize(n - 1LL, 0LL);
    edges.resize(n - 1LL);
    set<pair<long long, pair<long long, long long> > > s;
    for (i = 0; i < n - 1LL; i++) {
      cin >> j >> k >> l;
      --j, --k;
      if (j > k) swap(j, k);
      edges[i] = {j, k};
      adj[j].push_back({k, i});
      adj[k].push_back({j, i});
      mp[edges[i]] = l;
    }
    dfs(0LL);
    mx = 0LL;
    mn = 0LL;
    for (i = 0; i < n - 1; i++) {
      mx += mp[edges[i]] * cc[i];
      mn += mp[edges[i]] * (cc[i] % 2LL);
    }
    cout << mn << \" \" << mx << \"\n\";
    adj.clear();
    vis.clear();
    cc.clear();
    edges.clear();
    mp.clear();
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
import re
from typing import Dict, Any, Optional
from collections import defaultdict, deque

class Ejeremybearimybootcamp(Basebootcamp):
    def __init__(self, k: int = 3):
        """
        初始化训练场参数
        :param k: 灵魂伴侣对数
        """
        self.k = k

    def case_generator(self) -> Dict[str, Any]:
        """
        生成谜题实例
        :return: 包含树结构和答案的字典
        """
        n = 2 * self.k
        adj = [[] for _ in range(n)]
        edges = []
        root = 0
        visited = [False] * n
        q = deque([root])
        visited[root] = True

        while len(edges) < n - 1:
            u = random.choice(q)
            v = random.randint(0, n - 1)
            if not visited[v]:
                visited[v] = True
                adj[u].append(v)
                adj[v].append(u)
                q.append(v)
                weight = random.randint(1, 100)
                edges.append((u + 1, v + 1, weight))
            else:
                found = False
                while not found:
                    v = random.randint(0, n - 1)
                    if not visited[v] and v != u:
                        visited[v] = True
                        adj[u].append(v)
                        adj[v].append(u)
                        q.append(v)
                        weight = random.randint(1, 100)
                        edges.append((u + 1, v + 1, weight))
                        found = True

        edge_contributions = []

        def dfs(u, parent):
            size = 1
            for v in adj[u]:
                if v != parent:
                    child_size = dfs(v, u)
                    size += child_size
                    # Find the corresponding edge weight
                    for a, b, w in edges:
                        if (a == u + 1 and b == v + 1) or (a == v + 1 and b == u + 1):
                            edge_contributions.append((w, child_size))
                            break
            return size

        dfs(root, -1)

        G = 0
        B = 0
        for weight, cnt in edge_contributions:
            G += weight * (cnt % 2)
            B += weight * cnt

        case = {
            'k': self.k,
            'edges': edges,
            'G': G,
            'B': B
        }
        return case

    @staticmethod
    def prompt_func(question_case: Dict[str, Any]) -> str:
        """
        将问题实例转换为提示文本
        :param question_case: 生成的问题实例
        :return: 提示字符串
        """
        k = question_case['k']
        edges = question_case['edges']
        edge_list = "\n".join([f"House {a} connected to house {b} (time {t})" for a, b, t in edges])
        prompt = (
            "You are a soulmate assignment expert. Given a neighborhood of {total_houses} houses connected as a tree, "
            "you need to assign {pairs} pairs of soulmates to minimize and maximize the total travel time between each pair.\n"
            "The tree structure is:\n{edges}\n"
            "Compute two values:\n"
            "- G: Minimum possible total travel time\n"
            "- B: Maximum possible total travel time\n"
            "Please output your answer in the format:\n"
            "[answer]\nG = {G}, B = {B}\n[/answer]"
        ).format(
            total_houses=2 * k,
            pairs=k,
            edges=edge_list,
            G='G',
            B='B'
        )
        return prompt

    @staticmethod
    def extract_output(output: str) -> Optional[str]:
        """
        从模型输出中提取答案
        :param output: 模型输出的完整文本
        :return: 提取的答案字符串或None
        """
        match = re.search(r'\[answer\].*?G = (\d+), B = (\d+).*?\[/answer\]', output, re.DOTALL)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return None

    @classmethod
    def _verify_correction(cls, solution: str, identity: Dict[str, Any]) -> bool:
        """
        验证答案是否正确
        :param solution: 提取的答案字符串
        :param identity: 问题实例
        :return: 是否正确
        """
        try:
            G_str, B_str = solution.split()
            G = int(G_str)
            B = int(B_str)
            return G == identity['G'] and B == identity['B']
        except:
            return False

# 示例使用
if __name__ == "__main__":
    bootcamp = Ejeremybearimybootcamp(k=3)
    case = bootcamp.case_generator()
    print("Generated Case:", json.dumps(case, indent=2))
    prompt = Ejeremybearimybootcamp.prompt_func(case)
    print("\nPrompt:")
    print(prompt)
