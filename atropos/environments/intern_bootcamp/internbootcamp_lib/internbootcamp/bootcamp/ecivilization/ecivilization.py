"""# 

### 谜题描述
Andrew plays a game called \"Civilization\". Dima helps him.

The game has n cities and m bidirectional roads. The cities are numbered from 1 to n. Between any pair of cities there either is a single (unique) path, or there is no path at all. A path is such a sequence of distinct cities v1, v2, ..., vk, that there is a road between any contiguous cities vi and vi + 1 (1 ≤ i < k). The length of the described path equals to (k - 1). We assume that two cities lie in the same region if and only if, there is a path connecting these two cities.

During the game events of two types take place:

  1. Andrew asks Dima about the length of the longest path in the region where city x lies. 
  2. Andrew asks Dima to merge the region where city x lies with the region where city y lies. If the cities lie in the same region, then no merging is needed. Otherwise, you need to merge the regions as follows: choose a city from the first region, a city from the second region and connect them by a road so as to minimize the length of the longest path in the resulting region. If there are multiple ways to do so, you are allowed to choose any of them. 



Dima finds it hard to execute Andrew's queries, so he asks you to help him. Help Dima.

Input

The first line contains three integers n, m, q (1 ≤ n ≤ 3·105; 0 ≤ m < n; 1 ≤ q ≤ 3·105) — the number of cities, the number of the roads we already have and the number of queries, correspondingly.

Each of the following m lines contains two integers, ai and bi (ai ≠ bi; 1 ≤ ai, bi ≤ n). These numbers represent the road between cities ai and bi. There can be at most one road between two cities.

Each of the following q lines contains one of the two events in the following format:

  * 1 xi. It is the request Andrew gives to Dima to find the length of the maximum path in the region that contains city xi (1 ≤ xi ≤ n). 
  * 2 xi yi. It is the request Andrew gives to Dima to merge the region that contains city xi and the region that contains city yi (1 ≤ xi, yi ≤ n). Note, that xi can be equal to yi. 

Output

For each event of the first type print the answer on a separate line.

Examples

Input

6 0 6
2 1 2
2 3 4
2 5 6
2 3 2
2 5 3
1 1


Output

4

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 3e5 + 10;
int n, m, q;
int par[maxn], L[maxn];
int cnt[maxn];
void merge(int, int);
pair<int, int> root(int);
vector<int> adj[maxn];
void input() {
  scanf(\"%d%d%d\", &n, &m, &q);
  int v, u;
  for (int i = 0; i < n; i++) par[i] = -1;
  for (int i = 0; i < m; i++) {
    scanf(\"%d%d\", &u, &v);
    u--, v--;
    pair<int, int> Uinformation;
    adj[u].push_back(v);
    adj[v].push_back(u);
    merge(u, v);
  }
}
int dis[maxn];
bool mark[maxn];
int mx, ind;
void DFS_visit(int v) {
  mark[v] = 1;
  for (int i = 0; i < adj[v].size(); i++) {
    int nextV = adj[v][i];
    if (mark[nextV] == 0) {
      dis[nextV] = dis[v] + 1;
      if (mx < dis[nextV]) {
        mx = dis[nextV];
        ind = nextV;
      }
      mark[nextV] = 1;
      DFS_visit(nextV);
      dis[nextV] = 0;
      mark[nextV] = 0;
    }
  }
}
void DFS() {
  for (int i = 0; i < n; i++) {
    mx = 0;
    ind = -1;
    if (par[i] == -1) {
      DFS_visit(i);
      mx = 0;
      mark[i] = 0;
      dis[i] = 0;
      if (ind != -1) DFS_visit(ind);
      L[i] = mx;
    }
  }
}
pair<int, int> root(int v) {
  int now = v;
  int cnt = 0;
  while (par[now] != -1) {
    now = par[now];
    cnt++;
  }
  while (par[v] != -1) {
    int tmp = par[v];
    par[v] = now;
    v = tmp;
  }
  return make_pair(now, cnt);
}
void merge(int v, int u) {
  int parV = root(v).first, parU = root(u).first;
  if (parV == parU) return;
  if (cnt[parU] < cnt[parV]) {
    par[parU] = parV;
    cnt[parV] += cnt[parU];
    L[parV] = max(max(L[parV], L[parU]),
                  (int)ceil(L[parV] / 2.0) + (int)ceil(L[parU] / 2.0) + 1);
  } else {
    par[parV] = parU;
    cnt[parU] += cnt[parV];
    L[parU] = max(max(L[parV], L[parU]),
                  (int)ceil(L[parV] / 2.0) + (int)ceil(L[parU] / 2.0) + 1);
  }
}
int main() {
  input();
  DFS();
  int type, v, u;
  string s;
  for (int i = 0; i < q; i++) {
    scanf(\"%d\", &type);
    if (type == 1) {
      scanf(\"%d\", &v);
      cout << L[root(v - 1).first] << endl;
    } else {
      scanf(\"%d%d\", &v, &u);
      merge(v - 1, u - 1);
    }
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random

class Ecivilizationbootcamp(Basebootcamp):
    def __init__(self, default_n=6, default_m=0, default_q=6):
        self.default_n = default_n
        self.default_m = default_m  # 显式初始化m参数
        self.default_q = default_q  # 显式初始化q参数

    def case_generator(self):
        n = self.default_n
        m = self.default_m
        q = self.default_q

        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.diameter = [0] * size  # 正确初始化直径数组

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x, y):
                root_x = self.find(x)
                root_y = self.find(y)
                if root_x == root_y:
                    return
                # 使用正确的直径计算逻辑
                d1 = self.diameter[root_x]
                d2 = self.diameter[root_y]
                new_d = max(d1, d2, ((d1 + 1) // 2) + ((d2 + 1) // 2) + 1)
                self.parent[root_y] = root_x
                self.diameter[root_x] = new_d  # 保持直径更新逻辑一致

        uf = UnionFind(n)
        queries = []
        expected_outputs = []
        type1_count = 0
        
        # 强制至少生成一个类型1查询
        must_generate_type1 = True
        for _ in range(q):
            if must_generate_type1 or (random.random() < 0.3 and type1_count < q//2):
                x = random.randint(1, n)
                queries.append({'type': 1, 'x': x})
                root = uf.find(x-1)
                expected_outputs.append(uf.diameter[root])
                type1_count += 1
                must_generate_type1 = False  # 已满足最少一个
            else:
                x = random.randint(1, n)
                y = random.randint(1, n)
                queries.append({'type': 2, 'x': x, 'y': y})
                uf.union(x-1, y-1)

        case = {
            'n': n,
            'm': m,
            'q': q,
            'roads': [],  # 显式初始化空道路列表
            'queries': queries,
            'expected_outputs': expected_outputs
        }
        return case  # 确保返回数据结构完整

    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['m']} {question_case['q']}"]
        input_lines.extend([f"{q['x']} {q['y']}" for q in question_case['roads']])  # 正确遍历道路数据
        for query in question_case['queries']:
            if query['type'] == 1:
                input_lines.append(f"1 {query['x']}")
            else:
                input_lines.append(f"2 {query['x']} {query['y']}")
        
        prompt = (
            "在文明游戏中处理城市合并与路径长度查询，规则：\n"
            "1. 类型1查询返回城市所在区域最长路径\n"
            "2. 类型2合并不同区域时选择最优连接方式\n\n"
            "输入格式说明：\n"
            f"{chr(10).join(input_lines)}\n\n"
            "请将类型1查询结果按顺序放在[answer]标签内，如：\n"
            "[answer]\n3\n5\n[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        answers = []
        for line in answer_blocks[-1].splitlines():
            line = line.strip()
            if line and line.isdigit():
                answers.append(int(line))
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('expected_outputs', [])
