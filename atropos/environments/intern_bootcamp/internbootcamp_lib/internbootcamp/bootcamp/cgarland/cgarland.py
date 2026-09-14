"""# 

### 谜题描述
Once at New Year Dima had a dream in which he was presented a fairy garland. A garland is a set of lamps, some pairs of which are connected by wires. Dima remembered that each two lamps in the garland were connected directly or indirectly via some wires. Furthermore, the number of wires was exactly one less than the number of lamps.

There was something unusual about the garland. Each lamp had its own brightness which depended on the temperature of the lamp. Temperatures could be positive, negative or zero. Dima has two friends, so he decided to share the garland with them. He wants to cut two different wires so that the garland breaks up into three parts. Each part of the garland should shine equally, i. e. the sums of lamps' temperatures should be equal in each of the parts. Of course, each of the parts should be non-empty, i. e. each part should contain at least one lamp.

<image>

Help Dima to find a suitable way to cut the garland, or determine that this is impossible.

While examining the garland, Dima lifted it up holding by one of the lamps. Thus, each of the lamps, except the one he is holding by, is now hanging on some wire. So, you should print two lamp ids as the answer which denote that Dima should cut the wires these lamps are hanging on. Of course, the lamp Dima is holding the garland by can't be included in the answer.

Input

The first line contains single integer n (3 ≤ n ≤ 106) — the number of lamps in the garland.

Then n lines follow. The i-th of them contain the information about the i-th lamp: the number lamp ai, it is hanging on (and 0, if is there is no such lamp), and its temperature ti ( - 100 ≤ ti ≤ 100). The lamps are numbered from 1 to n.

Output

If there is no solution, print -1.

Otherwise print two integers — the indexes of the lamps which mean Dima should cut the wires they are hanging on. If there are multiple answers, print any of them.

Examples

Input

6
2 4
0 5
4 2
2 1
1 1
4 2


Output

1 4


Input

6
2 4
0 6
4 2
2 1
1 1
4 2


Output

-1

Note

The garland and cuts scheme for the first example:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n, nh[1000005], sum[1000005], t[1000005], p, root;
pair<int, int> kq;
vector<pair<int, int> > a[1000005];
void home() {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
}
void dfs(int u, int p) {
  sum[u] += t[u];
  if (a[u].size())
    for (int j = (0); j <= (a[u].size() - 1); ++j) {
      int v = a[u][j].first;
      if (v == p) continue;
      dfs(v, u);
      sum[u] += sum[v];
    }
}
void cal(int u, int p) {
  vector<int> vc;
  if (a[u].size())
    for (int j = (0); j <= (a[u].size() - 1); ++j) {
      int v = a[u][j].first;
      if (v == p) continue;
      cal(v, u);
      if (sum[v] == sum[root] / 3 && !nh[v]) nh[v] = a[u][j].second;
      if (nh[v]) vc.push_back(nh[v]);
      if (sum[v] == sum[root] - sum[root] / 3 && nh[v] &&
          nh[v] != a[u][j].second)
        kq = pair<int, int>(a[u][j].second, nh[v]);
    }
  if (vc.size() >= 2) kq = pair<int, int>(vc[0], vc[1]);
  if (vc.size()) nh[u] = vc[0];
}
int main() {
  home();
  cin >> n;
  for (int i = (1); i <= (n); ++i) {
    cin >> p >> t[i];
    if (p == 0)
      root = i;
    else
      a[p].push_back(pair<int, int>(i, i)),
          a[i].push_back(pair<int, int>(p, i));
  }
  dfs(root, 0);
  if (sum[root] % 3)
    cout << -1;
  else {
    cal(root, 0);
    if (kq == pair<int, int>(0, 0))
      cout << -1;
    else
      cout << kq.first << \" \" << kq.second;
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict

class Cgarlandbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=6, variant='random'):
        super().__init__()
        self.min_n = min_n
        self.max_n = max_n
        self.variant = variant

    def _generate_tree(self, n):
        """生成合法的树结构"""
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        root = nodes[0]
        parent = {root: 0}
        children = defaultdict(list)
        
        # 保证生成合法树结构
        for node in nodes[1:]:
            candidates = [p for p in parent if p != node]
            p = random.choice(candidates)
            parent[node] = p
            children[p].append(node)
        
        return parent, children, root

    def case_generator(self):
        # 生成基础树结构
        n = random.randint(self.min_n, self.max_n)
        parent, children, root = self._generate_tree(n)
        
        # 生成温度值
        temp = {node: random.randint(-10, 10) for node in parent}
        
        # 处理案例类型
        if self.variant == 'solvable' or (self.variant == 'random' and random.random() > 0.5):
            # 强制保证存在解
            total = sum(temp.values())
            remainder = total % 3
            
            # 调整三个独立子树的温度
            candidates = []
            for node in children[root]:
                if len(children[node]) == 0:  # 叶子节点
                    candidates.append(node)
                else:
                    for child in children[node]:
                        if len(children[child]) == 0:
                            candidates.append(child)
            
            if len(candidates) >= 3:
                adjust_nodes = random.sample(candidates, 3)
                target_sum = (total - sum(temp[n] for n in adjust_nodes)) // 3
                for node in adjust_nodes:
                    temp[node] = target_sum
            else:
                # 创建三个独立子树
                new_nodes = [n+1 for n in range(n, n+3)]
                for node in new_nodes:
                    parent[node] = root
                    temp[node] = random.randint(-5, 5)
                n += 3
                total = sum(temp.values())
            
            # 二次调整确保总和被3整除
            total = sum(temp.values())
            remainder = total % 3
            if remainder != 0:
                delta = (3 - remainder) % 3
                temp[random.choice(list(temp))] += delta
        else:
            # 确保无解
            total = sum(temp.values())
            if total % 3 == 0:
                # 通过调整打破可解性
                temp[random.choice(list(temp))] += 1
        
        # 构建案例数据
        lamps = []
        for i in range(1, n+1):
            lamps.append({'ai': parent.get(i, 0), 'ti': temp[i]})
        
        return {
            'n': n,
            'lamps': lamps,
            'parent': parent,
            'root': root,
            'sum_total': sum(temp.values())
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['n'])]
        for lamp in question_case['lamps']:
            input_lines.append(f"{lamp['ai']} {lamp['ti']}")
        example_input = '\n'.join(input_lines)
        return f"""根据以下输入，找出可以分割成三个相等温度部分的两条线缆。答案格式示例：[answer]1 4[/answer]\n输入：\n{example_input}"""

    @staticmethod
    def extract_output(output):
        match = re.search(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not match:
            return None
        content = match.group(1).strip()
        if content == '-1':
            return [-1]
        try:
            return sorted(map(int, content.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 实现完整的验证逻辑
        def build_tree(data):
            parent_map = {i+1: lamp['ai'] for i, lamp in enumerate(data['lamps'])}
            temp_map = {i+1: lamp['ti'] for i, lamp in enumerate(data['lamps'])}
            adj = defaultdict(list)
            for child in parent_map:
                parent = parent_map[child]
                if parent != 0:
                    adj[parent].append(child)
                    adj[child].append(parent)
            return adj, temp_map, data['root']

        def compute_subtree_sums(root, adj, temp_map):
            sums = {}
            def dfs(node, parent):
                total = temp_map[node]
                for child in adj[node]:
                    if child != parent:
                        total += dfs(child, node)
                sums[node] = total
                return total
            dfs(root, -1)
            return sums

        # 主验证逻辑
        if solution == [-1]:
            total = identity['sum_total']
            if total % 3 != 0:
                return True
            
            adj, temp_map, root = build_tree(identity)
            sums = compute_subtree_sums(root, adj, temp_map)
            
            # 寻找符合条件的子树
            candidates = [node for node in sums if sums[node] == total//3 and node != root]
            seen = set()
            for a in candidates:
                for b in candidates:
                    if a < b and not cls._is_ancestor(a, b, identity['parent']) and not cls._is_ancestor(b, a, identity['parent']):
                        return False
            return True
        else:
            # 验证正解逻辑
            if len(solution) != 2 or solution[0] == solution[1]:
                return False
            
            a, b = solution
            if identity['parent'].get(a, 0) == 0 or identity['parent'].get(b, 0) == 0:
                return False
            
            adj, temp_map, root = build_tree(identity)
            sums = compute_subtree_sums(root, adj, temp_map)
            
            total = sums[root]
            if total % 3 != 0:
                return False
            target = total // 3
            
            return (sums.get(a, -1) == target and 
                    sums.get(b, -1) == target and 
                    not cls._is_ancestor(a, b, identity['parent']) and 
                    not cls._is_ancestor(b, a, identity['parent']))

    @staticmethod
    def _is_ancestor(a, b, parent_map):
        current = b
        while current in parent_map:
            current = parent_map[current]
            if current == a:
                return True
        return False
