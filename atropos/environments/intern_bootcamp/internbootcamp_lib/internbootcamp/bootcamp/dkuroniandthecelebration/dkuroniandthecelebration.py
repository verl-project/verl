"""# 

### 谜题描述
This is an interactive problem.

After getting AC after 13 Time Limit Exceeded verdicts on a geometry problem, Kuroni went to an Italian restaurant to celebrate this holy achievement. Unfortunately, the excess sauce disoriented him, and he's now lost!

The United States of America can be modeled as a tree (why though) with n vertices. The tree is rooted at vertex r, wherein lies Kuroni's hotel.

Kuroni has a phone app designed to help him in such emergency cases. To use the app, he has to input two vertices u and v, and it'll return a vertex w, which is the lowest common ancestor of those two vertices.

However, since the phone's battery has been almost drained out from live-streaming Kuroni's celebration party, he could only use the app at most ⌊ n/2 ⌋ times. After that, the phone would die and there will be nothing left to help our dear friend! :(

As the night is cold and dark, Kuroni needs to get back, so that he can reunite with his comfy bed and pillow(s). Can you help him figure out his hotel's location?

Interaction

The interaction starts with reading a single integer n (2 ≤ n ≤ 1000), the number of vertices of the tree.

Then you will read n-1 lines, the i-th of them has two integers x_i and y_i (1 ≤ x_i, y_i ≤ n, x_i ≠ y_i), denoting there is an edge connecting vertices x_i and y_i. It is guaranteed that the edges will form a tree.

Then you can make queries of type \"? u v\" (1 ≤ u, v ≤ n) to find the lowest common ancestor of vertex u and v.

After the query, read the result w as an integer.

In case your query is invalid or you asked more than ⌊ n/2 ⌋ queries, the program will print -1 and will finish interaction. You will receive a Wrong answer verdict. Make sure to exit immediately to avoid getting other verdicts.

When you find out the vertex r, print \"! r\" and quit after that. This query does not count towards the ⌊ n/2 ⌋ limit.

Note that the tree is fixed beforehand and will not change during the queries, i.e. the interactor is not adaptive.

After printing any query do not forget to print end of line and flush the output. Otherwise, you might get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++;
  * System.out.flush() in Java;
  * flush(output) in Pascal;
  * stdout.flush() in Python;
  * see the documentation for other languages.



Hacks

To hack, use the following format:

The first line should contain two integers n and r (2 ≤ n ≤ 1000, 1 ≤ r ≤ n), denoting the number of vertices and the vertex with Kuroni's hotel.

The i-th of the next n-1 lines should contain two integers x_i and y_i (1 ≤ x_i, y_i ≤ n) — denoting there is an edge connecting vertex x_i and y_i.

The edges presented should form a tree.

Example

Input


6
1 4
4 2
5 3
6 3
2 3

3

4

4



Output








? 5 6

? 3 1

? 1 2

! 4

Note

Note that the example interaction contains extra empty lines so that it's easier to read. The real interaction doesn't contain any empty lines and you shouldn't print any extra empty lines as well.

The image below demonstrates the tree in the sample test:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <class T>
inline bool chmax(T &a, T b) {
  if (a < b) {
    a = b;
    return true;
  }
  return false;
}
template <class T>
inline bool chmin(T &a, T b) {
  if (a > b) {
    a = b;
    return true;
  }
  return false;
}
const long double EPS = 1e-10;
const long long INF = 1e18;
const long double PI = acos(-1.0L);
vector<int> paths[1005];
int N;
int ret;
void query(int a, int b) {
  cout << \"? \" << a << \" \" << b << endl;
  cin >> ret;
}
vector<int> v;
void dfs(int now, int from) {
  v.push_back(now);
  for (auto to : paths[now]) {
    if (to == from) continue;
    dfs(to, now);
  }
}
bool CUT(int root, int now, int from, int t) {
  if (now == t) return true;
  for (auto to : paths[now]) {
    if (to == from) continue;
    if (CUT(root, to, now, t)) {
      if (root == now) {
        auto itr = lower_bound(paths[now].begin(), paths[now].end(), to);
        paths[now].erase(itr);
      }
      return true;
    }
  }
  return false;
}
bool END = false;
void solve(int root) {
  v.clear();
  dfs(root, -1);
  if (v.size() == 1) {
    cout << \"! \" << root << endl;
    END = true;
    return;
  }
  if (v.size() == 2) {
    query(v[0], v[1]);
    cout << \"! \" << ret << endl;
    END = true;
    return;
  }
  bool ok = false;
  for (int i = 0; i < v.size(); i++) {
    if (ok) break;
    for (int j = 0; j < i; j++) {
      if (ok) break;
      auto itr = lower_bound(paths[v[i]].begin(), paths[v[i]].end(), v[j]);
      if (itr == paths[v[i]].end() or *itr != v[j]) {
        query(v[i], v[j]);
        if (ret == v[i]) {
          CUT(v[i], v[i], -1, v[j]);
        }
        if (ret == v[j]) {
          CUT(v[j], v[j], -1, v[i]);
        }
        if (ret != v[i] and ret != v[j]) {
          CUT(ret, ret, -1, v[i]);
          CUT(ret, ret, -1, v[j]);
        }
        ok = true;
      }
    }
  }
}
int main() {
  cin >> N;
  for (int i = 0; i < N - 1; i++) {
    int a, b;
    cin >> a >> b;
    paths[a].push_back(b);
    paths[b].push_back(a);
  }
  for (int i = 1; i <= N; i++) {
    sort(paths[i].begin(), paths[i].end());
  }
  ret = 1;
  while (true) {
    solve(ret);
    if (END) break;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Dkuroniandthecelebrationbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        初始化训练场类，设置默认参数以保存谜题相关参数。
        """
        self.n = params.get('n', random.randint(2, 1000))
        self.r = params.get('r', random.randint(1, self.n))
        self.edges = []
    
    def case_generator(self):
        """
        生成谜题实例，返回一个包含树结构和根节点的字典。
        """
        # 生成一个随机树
        self.n = random.randint(2, 1000)
        self.r = random.randint(1, self.n)
        self.edges = self.generate_tree(self.n, self.r)
        return {
            'n': self.n,
            'edges': self.edges.copy(),
            'root': self.r
        }
    
    def generate_tree(self, n, root):
        """
        生成一棵树的边列表，确保根节点为root。
        """
        edges = []
        nodes = list(range(1, n + 1))
        nodes.remove(root)
        # 使用随机方式生成树，确保根节点连接到所有子节点
        from collections import deque
        visited = set()
        visited.add(root)
        queue = deque()
        queue.append(root)
        while nodes and queue:
            u = queue.popleft()
            if nodes:
                v = random.choice(nodes)
                edges.append((u, v))
                visited.add(v)
                nodes.remove(v)
                queue.append(v)
        return edges
    
    @staticmethod
    def prompt_func(question_case):
        """
        将问题实例转换为文本形式的问题描述。
        """
        n = question_case['n']
        edges = question_case['edges']
        edge_strings = [f"{x} {y}" for x, y in edges]
        edge_list = "\n".join(edge_strings)
        prompt = (
            f"给定一棵包含{n}个节点的树，边如下所示：\n\n{edge_list}\n\n"
            "你需要通过查询两个节点的LCA来找出这棵树的根节点r。"
            f"每次查询的次数不能超过⌊{n}/2⌋次。\n"
            "找到根节点后，请输出它，放在[answer]标签中。例如：[answer]4[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        从LLM的回复中提取符合格式的答案。
        """
        import re
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output)
        if not matches:
            return None
        return matches[-1].strip()
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证提取的答案是否正确。
        """
        try:
            solution_int = int(solution)
            return solution_int == identity['root']
        except:
            return False
