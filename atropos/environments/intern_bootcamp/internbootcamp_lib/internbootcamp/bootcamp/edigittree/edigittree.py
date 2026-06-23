"""# 

### 谜题描述
ZS the Coder has a large tree. It can be represented as an undirected connected graph of n vertices numbered from 0 to n - 1 and n - 1 edges between them. There is a single nonzero digit written on each edge.

One day, ZS the Coder was bored and decided to investigate some properties of the tree. He chose a positive integer M, which is coprime to 10, i.e. <image>.

ZS consider an ordered pair of distinct vertices (u, v) interesting when if he would follow the shortest path from vertex u to vertex v and write down all the digits he encounters on his path in the same order, he will get a decimal representaion of an integer divisible by M.

Formally, ZS consider an ordered pair of distinct vertices (u, v) interesting if the following states true:

  * Let a1 = u, a2, ..., ak = v be the sequence of vertices on the shortest path from u to v in the order of encountering them; 
  * Let di (1 ≤ i < k) be the digit written on the edge between vertices ai and ai + 1; 
  * The integer <image> is divisible by M. 



Help ZS the Coder find the number of interesting pairs!

Input

The first line of the input contains two integers, n and M (2 ≤ n ≤ 100 000, 1 ≤ M ≤ 109, <image>) — the number of vertices and the number ZS has chosen respectively.

The next n - 1 lines contain three integers each. i-th of them contains ui, vi and wi, denoting an edge between vertices ui and vi with digit wi written on it (0 ≤ ui, vi < n, 1 ≤ wi ≤ 9).

Output

Print a single integer — the number of interesting (by ZS the Coder's consideration) pairs.

Examples

Input

6 7
0 1 2
4 2 4
2 0 1
3 0 9
2 5 7


Output

7


Input

5 11
1 2 3
2 0 3
3 0 3
4 3 3


Output

8

Note

In the first sample case, the interesting pairs are (0, 4), (1, 2), (1, 5), (3, 2), (2, 5), (5, 2), (3, 5). The numbers that are formed by these pairs are 14, 21, 217, 91, 7, 7, 917 respectively, which are all multiples of 7. Note that (2, 5) and (5, 2) are considered different. 

<image>

In the second sample case, the interesting pairs are (4, 0), (0, 4), (3, 2), (2, 3), (0, 1), (1, 0), (4, 1), (1, 4), and 6 of these pairs give the number 33 while 2 of them give the number 3333, which are all multiples of 11.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MIF = 1e9 + 7;
const double PI = 4 * atan(1);
inline long long in() {
  long long x = 0;
  int n = 1;
  char c = getchar();
  while (!isdigit(c)) {
    n = (c == '-') ? -1 : 1;
    c = getchar();
  }
  while (isdigit(c)) {
    x = x * 10 + c - '0';
    c = getchar();
  }
  return x * n;
}
inline char inc() {
  char c = getchar();
  while (!isalnum(c)) c = getchar();
  return c;
}
inline string ins() {
  string s = \"\";
  char c = getchar();
  while (!isalnum(c)) c = getchar();
  while (isalnum(c)) s = s + c, c = getchar();
  return s;
}
inline void out(long long x) {
  if (x < 0) putchar('-'), x = -x;
  if (x >= 10) out(x / 10);
  putchar(x % 10 + '0');
}
void fre() {
  ofstream fre(\"42.inp\");
  {
    fre << R\"(
6 7
0 1 2
4 2 4
2 0 1
3 0 9
2 5 7
)\";
  }
  fre.close();
  freopen(
      \"42\"
      \".inp\",
      \"r\", stdin);
}
const int N = 100005;
typedef long long ar_N[N];
ar_N wei, dep, di, ve, bac10;
long long PHI, res;
int n, M;
vector<pair<int, int> > g[N];
bool ct[N];
map<long long, int> mp, mp2, null;
void DFS_WEI(int first, int last) {
  wei[first] = 1;
  for (auto& i : g[first])
    if (!ct[i.first] && i.first != last) {
      DFS_WEI(i.first, first);
      wei[first] += wei[i.first];
    }
}
int CT_FIND(int first) {
  int halfwei = wei[first] / 2;
  while (1) {
    int last = first;
    for (auto& i : g[first])
      if (!ct[i.first] && halfwei < wei[i.first] && wei[i.first] < wei[first]) {
        first = i.first;
        break;
      }
    if (first == last) return first;
  }
}
long long POW(int first, long long second) {
  long long res = 1, xx = first;
  while (second > 0) {
    if (second & 1) res = (res * xx) % M;
    second >>= 1;
    xx = (xx * xx) % M;
  }
  return res;
}
void GET(long long Di, long long Dep) {
  Di = -Di;
  while (Di < 0) Di += M;
  Dep = POW(PHI, Dep);
  res += mp[(Di * Dep) % M] - mp2[(Di * Dep) % M];
}
void CT_CAL(int first, int last) {
  for (auto& i : g[first])
    if (!ct[i.first] && i.first != last) {
      dep[i.first] = dep[first] + 1;
      ve[i.first] = ((i.second * bac10[dep[first]]) % M + ve[first]) % M;
      di[i.first] = (di[first] * 10 + i.second) % M;
      if (di[i.first] == 0) res++;
      if (ve[i.first] == 0) res++;
      mp[ve[i.first]]++;
      CT_CAL(i.first, first);
    }
}
void CT_GET(int first, int last) {
  GET(di[first], dep[first]);
  for (auto& i : g[first])
    if (!ct[i.first] && i.first != last) CT_GET(i.first, first);
}
void CT_UPDATE(int first, int last) {
  mp2[ve[first]]++;
  for (auto& i : g[first])
    if (!ct[i.first] && i.first != last) CT_UPDATE(i.first, first);
}
void CT_SOLVE(int first) {
  mp = null;
  DFS_WEI(first, first);
  first = CT_FIND(first);
  dep[first] = ve[first] = di[first] = 0;
  CT_CAL(first, first);
  for (auto& i : g[first])
    if (!ct[i.first]) {
      mp2 = null;
      CT_UPDATE(i.first, first);
      CT_GET(i.first, first);
    }
  ct[first] = true;
  for (auto& i : g[first])
    if (!ct[i.first]) CT_SOLVE(i.first);
}
void MAKE_PHI() {
  long long m = M;
  PHI = m;
  for (int i = 2; i * i <= m; i++)
    if (m % i == 0) {
      PHI = ((PHI / i) * (i - 1));
      while (m % i == 0) m /= i;
    }
  if (m > 1) PHI = ((PHI / m) * (m - 1));
  PHI--;
  PHI = POW(10, PHI);
}
int main() {
  n = in(), M = in();
  bac10[0] = 1;
  for (int i = 1; i <= 100000; i++) bac10[i] = (bac10[i - 1] * 10) % M;
  MAKE_PHI();
  for (int i_ = 1; i_ <= n - 1; i_++) {
    int first = in(), second = in(), z = in();
    g[first].emplace_back(second, z);
    g[second].emplace_back(first, z);
  }
  CT_SOLVE(1);
  out(res);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import math
from collections import deque
from itertools import combinations
from bootcamp import Basebootcamp

class Edigittreebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 2)
        self.n_max = params.get('n_max', 8)  # 适当提高上限仍保证暴力验证可行
        self.m_max = params.get('m_max', 100)
    
    def _valid_m_generator(self):
        """生成与10互质的M，包含边界情况处理"""
        candidates = [1]  # 特殊情况M=1
        for _ in range(100):
            m = random.randint(2, self.m_max)
            if math.gcd(m, 10) == 1:
                candidates.append(m)
        return random.choice(candidates)
    
    def _generate_tree(self, n):
        """更健壮的树生成算法，覆盖链状和星型结构"""
        if random.random() < 0.3:  # 30%概率生成链状树
            edges = []
            for i in range(1, n):
                edges.append((i-1, i, random.randint(1,9)))
            return edges
        else:  # 常规随机树
            edges = []
            nodes = list(range(n))
            random.shuffle(nodes)
            for i in range(1, n):
                u = random.choice(nodes[:i])
                v = nodes[i]
                edges.append((u, v, random.randint(1,9)))
            return edges
    
    def case_generator(self):
        # 生成符合要求的M
        M = self._valid_m_generator()
        n = random.randint(self.n_min, self.n_max)
        
        # 生成规范化的树结构
        edges = self._generate_tree(n)
        
        # 暴力验证答案
        def compute_expected(n_val, m_val, edges_list):
            # 构建邻接表
            adj = [[] for _ in range(n_val)]
            for u, v, w in edges_list:
                adj[u].append((v, w))
                adj[v].append((u, w))
            
            # 预处理所有节点对
            count = 0
            for u, v in combinations(range(n_val), 2):
                # 双向路径查找
                for src, dst in [(u, v), (v, u)]:
                    # BFS找路径
                    visited = {src: None}
                    q = deque([src])
                    while q:
                        node = q.popleft()
                        if node == dst:
                            break
                        for neighbor, weight in adj[node]:
                            if neighbor not in visited:
                                visited[neighbor] = (node, weight)
                                q.append(neighbor)
                    
                    # 提取路径数字
                    path = []
                    current = dst
                    while visited.get(current) is not None:
                        prev_node, weight = visited[current]
                        path.append(weight)
                        current = prev_node
                    num = 0
                    for digit in reversed(path):
                        num = num * 10 + digit
                        num %= m_val
                    if num % m_val == 0:
                        count += 1
            return count
        
        expected = compute_expected(n, M, edges)
        
        return {
            'n': n,
            'M': M,
            'edges': edges,
            'expected_answer': expected
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['M']}"]
        for edge in question_case['edges']:
            input_lines.append(f"{edge[0]} {edge[1]} {edge[2]}")
        input_str = '\n'.join(input_lines)
        
        prompt = f"""Given a tree with {question_case['n']} vertices where each edge contains a digit (1-9), find the number of ordered pairs (u, v), u≠v, such that the decimal number formed by the path from u to v is divisible by {question_case['M']}.

Input format:
First line: n M
Next n-1 lines: u v w (edge between u and v with digit w)

Example valid output format:
The answer is [answer]42[/answer] where 42 is the correct count.

Your Input:
{input_str}

Calculate the answer and put your final numerical answer within [answer] tags."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 强化抽取逻辑，处理多种数字格式
        matches = re.findall(r'\[answer\s*\]\s*(\d+)\s*\[/answer\s*\]', output, re.IGNORECASE)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 添加类型校验
        if not isinstance(solution, int):
            return False
        return solution == identity['expected_answer']
