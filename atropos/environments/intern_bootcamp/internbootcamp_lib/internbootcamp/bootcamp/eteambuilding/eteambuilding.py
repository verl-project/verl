"""# 

### 谜题描述
The new academic year has started, and Berland's university has n first-year students. They are divided into k academic groups, however, some of the groups might be empty. Among the students, there are m pairs of acquaintances, and each acquaintance pair might be both in a common group or be in two different groups.

Alice is the curator of the first years, she wants to host an entertaining game to make everyone know each other. To do that, she will select two different academic groups and then divide the students of those groups into two teams. The game requires that there are no acquaintance pairs inside each of the teams.

Alice wonders how many pairs of groups she can select, such that it'll be possible to play a game after that. All students of the two selected groups must take part in the game.

Please note, that the teams Alice will form for the game don't need to coincide with groups the students learn in. Moreover, teams may have different sizes (or even be empty).

Input

The first line contains three integers n, m and k (1 ≤ n ≤ 500 000; 0 ≤ m ≤ 500 000; 2 ≤ k ≤ 500 000) — the number of students, the number of pairs of acquaintances and the number of groups respectively.

The second line contains n integers c_1, c_2, ..., c_n (1 ≤ c_i ≤ k), where c_i equals to the group number of the i-th student.

Next m lines follow. The i-th of them contains two integers a_i and b_i (1 ≤ a_i, b_i ≤ n), denoting that students a_i and b_i are acquaintances. It's guaranteed, that a_i ≠ b_i, and that no (unordered) pair is mentioned more than once.

Output

Print a single integer — the number of ways to choose two different groups such that it's possible to select two teams to play the game.

Examples

Input


6 8 3
1 1 2 2 3 3
1 3
1 5
1 6
2 5
2 6
3 4
3 5
5 6


Output


2


Input


4 3 3
1 1 2 2
1 2
2 3
3 4


Output


3


Input


4 4 2
1 1 1 2
1 2
2 3
3 1
1 4


Output


0


Input


5 5 2
1 2 1 2 1
1 2
2 3
3 4
4 5
5 1


Output


0

Note

The acquaintances graph for the first example is shown in the picture below (next to each student there is their group number written).

<image>

In that test we can select the following groups:

  * Select the first and the second groups. For instance, one team can be formed from students 1 and 4, while other team can be formed from students 2 and 3. 
  * Select the second and the third group. For instance, one team can be formed 3 and 6, while other team can be formed from students 4 and 5. 
  * We can't select the first and the third group, because there is no way to form the teams for the game. 



In the second example, we can select any group pair. Please note, that even though the third group has no students, we still can select it (with some other group) for the game.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using std::cerr;
using std::cin;
using std::max;
using std::min;
int n, m, K, a, b, ecnt, last[500005], A[500005], Bn, f[500005 << 1],
    size[500005 << 1], mark[500005], sta[500005 << 1], top, tot;
long long ans;
struct node {
  int x, y;
} B[500005];
struct road {
  int to, nex;
} e[500005];
void adde(int u, int v) { e[++ecnt] = {v, last[u]}, last[u] = ecnt; }
int find(int x) { return f[x] ^ x ? find(f[x]) : x; }
bool merge(int x, int y) {
  int f1 = find(x), f2 = find(y);
  if (f1 ^ f2) {
    if (size[f1] > size[f2]) std::swap(f1, f2);
    f[f1] = f2, size[f2] += size[f1], sta[++top] = f1;
  }
  return find(x) == find(x + n);
}
void Solve(int l, int r) {
  if (mark[A[B[l].x]] || mark[A[B[l].y]]) return;
  int flag = 0;
  for (int i = l; i <= r && !flag; i++)
    a = B[i].x, b = B[i].y, flag |= merge(a, b + n) | merge(b, a + n);
  if (flag) ans++;
  for (int x; top;) x = sta[top--], size[f[x]] -= size[x], f[x] = x;
}
int main() {
  scanf(\"%d%d%d\", &n, &m, &K);
  for (int i = 1; i <= n; i++) scanf(\"%d\", &A[i]);
  for (int i = 1; i <= n << 1; i++) f[i] = i, size[i] = 1;
  for (int i = 1; i <= m; i++) {
    scanf(\"%d%d\", &a, &b);
    if (A[a] ^ A[b]) {
      if (A[a] > A[b]) std::swap(a, b);
      B[++Bn] = {a, b};
    } else
      mark[A[a]] |= merge(a, b + n) | merge(b, a + n);
  }
  for (int i = 1; i <= K; i++) tot += mark[i];
  ans = (long long)tot * (K - tot) + (long long)tot * (tot - 1) / 2, top = 0;
  std::sort(B + 1, B + 1 + Bn, [](node t1, node t2) {
    return A[t1.x] < A[t2.x] || A[t1.x] == A[t2.x] && A[t1.y] < A[t2.y];
  });
  for (int i = 1, j = 1; i <= Bn; i++)
    if (A[B[i].x] ^ A[B[i + 1].x] || A[B[i].y] ^ A[B[i + 1].y] || i == Bn)
      Solve(j, i), j = i + 1;
  printf(\"%lld\n\", (long long)K * (K - 1) / 2 - ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
import re
from bootcamp import Basebootcamp

class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size
        self.history = []
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x
    
    def merge(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx == fy:
            return
        if self.size[fx] < self.size[fy]:
            fx, fy = fy, fx
        self.history.append((fy, fx))  # 记录合并顺序
        self.parent[fy] = fx
        self.size[fx] += self.size[fy]
    
    def rollback(self, checkpoint):
        while len(self.history) > checkpoint:
            fy, fx = self.history.pop()
            self.parent[fy] = fy
            self.size[fx] -= self.size[fy]

def compute_correct_answer(n, m, k, c_list, edges):
    group_edges = defaultdict(list)
    cross_edges = []
    mark = defaultdict(bool)
    
    # 学生编号1-based处理
    c = [0] * (n + 1)
    for i in range(1, n+1):
        c[i] = c_list[i-1]
    
    dsu = DSU(2*(n+2))  # 每个节点分拆为两个
    
    # 分离同组边和跨组边
    for a, b in edges:
        if c[a] == c[b]:
            group_edges[c[a]].append((a, b))
        else:
            u, v = sorted([c[a], c[b]])
            cross_edges.append((u, v, a, b))
    
    # 处理同组边（标记矛盾组）
    for group in group_edges:
        conflict = False
        cp = len(dsu.history)
        for a, b in group_edges[group]:
            # 检查合并是否产生矛盾
            dsu.merge(a, b + n)
            dsu.merge(b, a + n)
            if dsu.find(a) == dsu.find(a + n):
                conflict = True
                break
        if conflict:
            mark[group] = True
        dsu.rollback(cp)  # 回滚到处理前的状态
    
    # 排序跨组边（关键修正点）
    cross_edges.sort(key=lambda x: (x[0], x[1]))
    
    # 统计无效组对
    total_pairs = k * (k - 1) // 2
    invalid_pairs = 0
    tot_marked = sum(mark.values())
    invalid_pairs += tot_marked * (k - tot_marked) + tot_marked * (tot_marked - 1) // 2
    
    # 处理跨组边（修正排序逻辑）
    i = 0
    while i < len(cross_edges):
        j = i
        current_u = cross_edges[i][0]
        current_v = cross_edges[i][1]
        while j < len(cross_edges) and cross_edges[j][0:2] == (current_u, current_v):
            j += 1
        
        if mark[current_u] or mark[current_v]:
            invalid_pairs += 1
            i = j
            continue
        
        conflict = False
        cp = len(dsu.history)
        for idx in range(i, j):
            _, _, a, b = cross_edges[idx]
            dsu.merge(a, b + n)
            dsu.merge(b, a + n)
            if dsu.find(a) == dsu.find(a + n) or dsu.find(b) == dsu.find(b + n):
                conflict = True
                break
        
        if conflict:
            invalid_pairs += 1
        dsu.rollback(cp)
        i = j
    
    return total_pairs - invalid_pairs

class Eteambuildingbootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_m=20, max_k=10):
        self.max_n = max_n  # 适当扩大测试规模
        self.max_m = max_m
        self.max_k = max_k
    
    def case_generator(self):
        k = random.randint(2, self.max_k)
        n = random.randint(1, self.max_n)
        
        # 允许生成空组（重要修正点）
        c_list = []
        groups = list(range(1, k+1)) + [random.randint(1, k) for _ in range(3)]  # 增加重复概率
        for _ in range(n):
            c_list.append(random.choice(groups))
        
        # 生成唯一边集
        edge_set = set()
        students = list(range(1, n+1))
        for _ in range(min(self.max_m, n*(n-1)//2)):
            a, b = random.sample(students, 2)
            a, b = sorted([a, b])
            edge_set.add((a, b))
        edges = list(edge_set)
        
        return {
            "n": n,
            "m": len(edges),
            "k": k,
            "c_list": c_list,
            "edges": edges,
            "correct_answer": compute_correct_answer(n, len(edges), k, c_list, edges)
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [
            f"{question_case['n']} {question_case['m']} {question_case['k']}",
            ' '.join(map(str, question_case['c_list']))
        ]
        input_lines.extend(f"{a} {b}" for a, b in question_case['edges'])
        input_str = '\n'.join(input_lines)
        prompt = f"""Alice需要选择两个不同的学术组，使得这两个组的所有学生可以被分成两队且每队内没有熟人。请根据以下输入数据计算有效的组对数量，答案格式为[answer]数字[/answer]。

输入格式：
n m k
c_1 c_2 ... c_n
a_1 b_1
...
a_m b_m

输入数据：
{input_str}

请将最终答案放在[answer]标签内。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 强化正则匹配（支持含空格的情况）
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
