"""# 

### 谜题描述
Limak is a little bear who learns to draw. People usually start with houses, fences and flowers but why would bears do it? Limak lives in the forest and he decides to draw a tree.

Recall that tree is a connected graph consisting of n vertices and n - 1 edges.

Limak chose a tree with n vertices. He has infinite strip of paper with two parallel rows of dots. Little bear wants to assign vertices of a tree to some n distinct dots on a paper so that edges would intersect only at their endpoints — drawn tree must be planar. Below you can see one of correct drawings for the first sample test.

<image>

Is it possible for Limak to draw chosen tree?

Input

The first line contains single integer n (1 ≤ n ≤ 105).

Next n - 1 lines contain description of a tree. i-th of them contains two space-separated integers ai and bi (1 ≤ ai, bi ≤ n, ai ≠ bi) denoting an edge between vertices ai and bi. It's guaranteed that given description forms a tree.

Output

Print \"Yes\" (without the quotes) if Limak can draw chosen tree. Otherwise, print \"No\" (without the quotes).

Examples

Input

8
1 2
1 3
1 6
6 4
6 7
6 5
7 8


Output

Yes


Input

13
1 2
1 3
1 4
2 5
2 6
2 7
3 8
3 9
3 10
4 11
4 12
4 13


Output

No

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n;
vector<vector<int> > adj;
vector<bool> top;
vector<int> branched;
vector<bool> visited;
vector<bool> fertile;
int root = -1;
void fail() {
  cout << \"No\" << endl;
  exit(0);
}
void dfsFindRoot(int u) {
  if (root != -1) return;
  visited[u] = true;
  bool leaf = true;
  for (int child : adj[u]) {
    if (visited[child]) continue;
    leaf = false;
  }
  if (!leaf) {
    branched[u] = 0;
    for (int child : adj[u]) {
      if (visited[child]) continue;
      dfsFindRoot(child);
      branched[u] += branched[child];
    }
    if (root != -1) return;
    if (branched[u] >= 3) {
      root = u;
    }
  }
}
void dfs(int u) {
  visited[u] = true;
  bool leaf = true;
  for (int child : adj[u]) {
    if (visited[child]) continue;
    leaf = false;
  }
  if (!leaf) {
    branched[u] = 0;
    int numTops = 0;
    for (int child : adj[u]) {
      if (visited[child]) continue;
      dfs(child);
      if (fertile[child]) top[u] = true;
      branched[u] += branched[child];
      if (top[child]) {
        numTops++;
        top[u] = true;
      }
    }
    if (u != root && numTops > 1)
      fail();
    else if (numTops > 2)
      fail();
    if (branched[u] >= 3) {
      top[u] = true;
    } else if (branched[u] == 2)
      fertile[u] = true;
  }
}
int main() {
  int m;
  cin >> n;
  adj = vector<vector<int> >(n, vector<int>(0));
  for (int i = 0; i < n - 1; i++) {
    int u, v;
    cin >> u >> v;
    u--;
    v--;
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  if (n <= 4) {
    cout << \"YES\" << endl;
    return 0;
  }
  top = vector<bool>(n, false);
  branched = vector<int>(n, 1);
  visited = vector<bool>(n, false);
  dfsFindRoot(0);
  if (root == -1) {
    cout << \"Yes\" << endl;
    return 0;
  }
  top = vector<bool>(n, false);
  branched = vector<int>(n, 1);
  visited = vector<bool>(n, false);
  fertile = vector<bool>(n, false);
  dfs(root);
  cout << \"Yes\" << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Ebearanddrawingbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        初始化谜题参数，默认n的范围为5-20
        """
        self.params = params
        self.params.setdefault('min_n', 5)
        self.params.setdefault('max_n', 20)
    
    def case_generator(self):
        """
        生成随机树结构并计算正确答案
        """
        n = random.randint(self.params['min_n'], self.params['max_n'])
        edges = self._generate_random_tree(n)
        correct_answer = self.is_tree_planar(n, edges)
        return {
            'n': n,
            'edges': edges,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def _generate_random_tree(n):
        """
        使用Prüfer序列生成随机树
        """
        if n == 1:
            return []
        if n == 2:
            return [(1, 2)]
        prufer = [random.randint(1, n) for _ in range(n-2)]
        degree = [1] * (n + 1)
        
        for node in prufer:
            degree[node] += 1
        
        edges = []
        for node in prufer:
            for v in range(1, n+1):
                if degree[v] == 1:
                    edges.append((node, v))
                    degree[node] -= 1
                    degree[v] -= 1
                    break
        
        u = [i for i in range(1, n+1) if degree[i] == 1]
        edges.append((u[0], u[1]))
        return edges
    
    @staticmethod
    def is_tree_planar(n, edges):
        """
        判断树是否可以被平面绘制
        """
        if n <= 4:
            return "Yes"
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u-1].append(v-1)
            adj[v-1].append(u-1)
        
        root = -1
        visited = [False] * n
        branched = [1] * n
        
        def dfs_find_root(u):
            nonlocal root
            if root != -1:
                return
            visited[u] = True
            has_child = False
            for child in adj[u]:
                if not visited[child]:
                    has_child = True
                    dfs_find_root(child)
                    branched[u] += branched[child]
            if has_child and branched[u] >= 3:
                root = u
        
        dfs_find_root(0)
        if root == -1:
            return "Yes"
        
        top = [False] * n
        branched = [1] * n
        visited = [False] * n
        fertile = [False] * n
        failed = False
        
        def dfs(u):
            nonlocal failed
            if failed:
                return
            visited[u] = True
            leaf = True
            for child in adj[u]:
                if not visited[child]:
                    leaf = False
            if not leaf:
                branched[u] = 0
                num_tops = 0
                for child in adj[u]:
                    if visited[child]:
                        continue
                    dfs(child)
                    if fertile[child]:
                        top[u] = True
                    branched[u] += branched[child]
                    if top[child]:
                        num_tops += 1
                        top[u] = True
                if (u != root and num_tops > 1) or (u == root and num_tops > 2):
                    failed = True
                if branched[u] >= 3:
                    top[u] = True
                elif branched[u] == 2:
                    fertile[u] = True
        
        visited = [False] * n
        dfs(root)
        return "Yes" if not failed else "No"
    
    @staticmethod
    def prompt_func(question_case):
        """
        生成格式化的谜题问题
        """
        n = question_case['n']
        edges = '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        problem = f"""Limak wants to draw a tree on a special paper strip with two rows of dots. The tree must be drawn with no edge crossings except at vertices. Determine if this is possible for the given tree structure.

Input:
{n}
{edges}

Output your answer exactly as "[answer]Yes[/answer]" or "[answer]No[/answer]" after analyzing the structure."""
        return problem
    
    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取最后一个[answer]标签内容
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
        answer = matches[-1].strip().lower()
        return answer.capitalize() if answer in {'yes', 'no'} else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案是否正确
        """
        return solution == identity['correct_answer']
