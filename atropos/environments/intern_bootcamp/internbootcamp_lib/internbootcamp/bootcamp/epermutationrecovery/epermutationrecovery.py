"""# 

### 谜题描述
Vasya has written some permutation p_1, p_2, …, p_n of integers from 1 to n, so for all 1 ≤ i ≤ n it is true that 1 ≤ p_i ≤ n and all p_1, p_2, …, p_n are different. After that he wrote n numbers next_1, next_2, …, next_n. The number next_i is equal to the minimal index i < j ≤ n, such that p_j > p_i. If there is no such j let's let's define as next_i = n + 1.

In the evening Vasya went home from school and due to rain, his notebook got wet. Now it is impossible to read some written numbers. Permutation and some values next_i are completely lost! If for some i the value next_i is lost, let's say that next_i = -1.

You are given numbers next_1, next_2, …, next_n (maybe some of them are equal to -1). Help Vasya to find such permutation p_1, p_2, …, p_n of integers from 1 to n, that he can write it to the notebook and all numbers next_i, which are not equal to -1, will be correct. 

Input

The first line contains one integer t — the number of test cases (1 ≤ t ≤ 100 000).

Next 2 ⋅ t lines contains the description of test cases,two lines for each. The first line contains one integer n — the length of the permutation, written by Vasya (1 ≤ n ≤ 500 000). The second line contains n integers next_1, next_2, …, next_n, separated by spaces (next_i = -1 or i < next_i ≤ n + 1).

It is guaranteed, that the sum of n in all test cases doesn't exceed 500 000.

In hacks you can only use one test case, so T = 1.

Output

Print T lines, in i-th of them answer to the i-th test case.

If there is no such permutations p_1, p_2, …, p_n of integers from 1 to n, that Vasya could write, print the only number -1.

In the other case print n different integers p_1, p_2, …, p_n, separated by spaces (1 ≤ p_i ≤ n). All defined values of next_i which are not equal to -1 should be computed correctly p_1, p_2, …, p_n using defenition given in the statement of the problem. If there exists more than one solution you can find any of them.

Example

Input


6
3
2 3 4
2
3 3
3
-1 -1 -1
3
3 4 -1
1
2
4
4 -1 4 5


Output


1 2 3
2 1
2 1 3
-1
1
3 2 1 4

Note

In the first test case for permutation p = [1, 2, 3] Vasya should write next = [2, 3, 4], because each number in permutation is less than next. It's easy to see, that it is the only satisfying permutation.

In the third test case, any permutation can be the answer because all numbers next_i are lost.

In the fourth test case, there is no satisfying permutation, so the answer is -1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
int n, nxt[500500], g[500500][2], deg[500500], o[500500], s[500500], in[500500],
    counter;
bool ok;
void topo(int x) {
  if (x < 0 || x >= n) return;
  if (o[x] > 0) return;
  if (o[x] == -1) {
    ok = false;
    return;
  }
  o[x] = -1;
  for (ll i = (0); i < (deg[x]); ++i) topo(g[x][i]);
  o[x] = counter--;
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  int t;
  cin >> t;
  while (t--) {
    cin >> n;
    counter = n;
    ok = true;
    for (ll i = (0); i < (n); ++i) {
      o[i] = 0;
      cin >> nxt[i];
      if (nxt[i] > 0) --nxt[i];
    }
    for (ll i = (0); i < (n); ++i) g[i][0] = nxt[i], deg[i] = 1;
    int top = -1;
    for (ll i = (0); i < (n); ++i) {
      while (top >= 0 && nxt[s[top]] <= i) --top;
      if (top >= 0) g[i][1] = s[top], deg[i] = 2;
      if (nxt[i] >= 0) s[++top] = i;
    }
    for (ll i = (0); i < (n); ++i) in[i] = 0;
    for (ll i = (0); i < (n); ++i)
      for (ll j = (0); j < (deg[i]); ++j) ++in[g[i][j]];
    for (ll i = (0); i < (n); ++i)
      if (in[i] == 0) topo(i);
    for (ll i = (0); i < (n); ++i)
      if (!o[i]) ok = false;
    if (!ok) {
      cout << -1 << endl;
    } else {
      for (ll i = (0); i < (n); ++i) {
        if (i) cout << \" \";
        cout << o[i];
      }
      cout << endl;
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque

class Epermutationrecoverybootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_range = params.get('n_range', (1, 10))
        self.mask_prob = params.get('mask_prob', 0.4)
        self.unsolve_prob = params.get('unsolve_prob', 0.3)

    def case_generator(self):
        if random.random() < self.unsolve_prob:
            return self._generate_unsolvable_case()
        return self._generate_solvable_case()

    def _generate_solvable_case(self):
        n = random.randint(*self.n_range)
        p = list(range(1, n+1))
        random.shuffle(p)
        next_list = self.compute_next(p)
        masked_next = [
            x if random.random() < self.mask_prob else -1
            for x in next_list
        ]
        return {'n': n, 'next': masked_next}

    def _generate_unsolvable_case(self):
        conflict_types = [
            self._create_cycle_conflict,
            self._create_order_conflict,
            self._create_range_conflict
        ]
        for _ in range(50):
            creator = random.choice(conflict_types)
            case = creator()
            if case and not self.check_solvable(case['n'], case['next']):
                return case
        return {'n': 3, 'next': [3, 4, -1]}

    def _create_cycle_conflict(self):
        n = random.randint(3, 6)
        next_list = [-1]*n
        for i in range(n-1):
            next_list[i] = i+2  # 创建循环依赖
        next_list[-1] = 1
        return {'n': n, 'next': next_list}

    def _create_order_conflict(self):
        n = random.randint(4, 6)
        next_list = [-1]*n
        next_list[0] = n+1  # 无效的next值
        for i in range(1, n-1):
            next_list[i] = i+2
        return {'n': n, 'next': next_list}

    def _create_range_conflict(self):
        n = 5
        return {'n': n, 'next': [3, 6, 4, 6, -1]}

    @staticmethod
    def compute_next(p):
        n = len(p)
        next_arr = []
        for i in range(n):
            min_j = n + 1
            for j in range(i+1, n):
                if p[j] > p[i]:
                    min_j = j + 1
                    break
            next_arr.append(min_j)
        return next_arr

    @staticmethod
    def check_solvable(n, next_list):
        next_array = [x-1 if x != -1 else -1 for x in next_list]
        graph = [[] for _ in range(n)]
        stack = []

        # 构建图结构
        for i in range(n):
            if 0 <= next_array[i] < n:
                graph[i].append(next_array[i])

            while stack and (next_array[stack[-1]] == -1 or next_array[stack[-1]] <= i):
                stack.pop()
            if stack:
                graph[i].append(stack[-1])
            if next_array[i] != -1 and next_array[i] != n:
                stack.append(i)

        # 拓扑排序检测
        in_degree = [0]*n
        for u in range(n):
            for v in graph[u]:
                if 0 <= v < n:
                    in_degree[v] += 1

        queue = deque([u for u in range(n) if in_degree[u] == 0])
        visited = 0

        while queue:
            u = queue.popleft()
            visited += 1
            for v in graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        return visited == n

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        next_values = ' '.join(map(str, question_case['next']))
        return f"""根据给定的部分next值，重建合法排列。规则：
1. next_i表示i之后第一个大于p_i的索引（1-based）
2. 若无更大元素则设为{n+1}
3. 丢失值用-1表示

输入：
n = {n}
next = [{next_values}]

请将最终答案用[answer]包裹，示例：
[answer]3 1 2[/answer] 或 [answer]-1[/answer]"""

    @staticmethod
    def extract_output(output):
        patterns = [
            r'\[answer\]\s*(-?\d[\d\s]*?)\s*\[/answer\]',
            r'answer:\s*(-?\d[\d\s]*)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            if matches:
                last = matches[-1].strip()
                try:
                    return -1 if last == '-1' else list(map(int, last.split()))
                except:
                    continue
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证无解情况
        if solution == -1:
            return not cls.check_solvable(identity['n'], identity['next'])
        
        # 验证排列格式
        n = identity['n']
        if len(solution) != n or set(solution) != set(range(1, n+1)):
            return False

        # 验证next匹配
        expected_next = cls.compute_next(solution)
        for given, actual in zip(identity['next'], expected_next):
            if given != -1 and given != actual:
                return False
        return True
