"""# 

### 谜题描述
Rainbow built h cells in a row that are numbered from 1 to h from left to right. There are n cells with treasure. We call each of these n cells \"Treasure Cell\". The i-th \"Treasure Cell\" is the ai-th cell and the value of treasure in it is ci dollars.

Then, Freda went in the first cell. For now, she can go just k cells forward, or return to the first cell. That means Freda was able to reach the 1st, (k + 1)-th, (2·k + 1)-th, (3·k + 1)-th cells and so on.

Then Rainbow gave Freda m operations. Each operation is one of the following three types:

  1. Add another method x: she can also go just x cells forward at any moment. For example, initially she has only one method k. If at some moment she has methods a1, a2, ..., ar then she can reach all the cells with number in form <image>, where vi — some non-negative integer. 
  2. Reduce the value of the treasure in the x-th \"Treasure Cell\" by y dollars. In other words, to apply assignment cx = cx - y. 
  3. Ask the value of the most valuable treasure among the cells Freda can reach. If Freda cannot reach any cell with the treasure then consider the value of the most valuable treasure equal to 0, and do nothing. Otherwise take the most valuable treasure away. If several \"Treasure Cells\" have the most valuable treasure, take the \"Treasure Cell\" with the minimum number (not necessarily with the minimum number of cell). After that the total number of cells with a treasure is decreased by one. 



As a programmer, you are asked by Freda to write a program to answer each query.

Input

The first line of the input contains four integers: h (1 ≤ h ≤ 1018), n, m (1 ≤ n, m ≤ 105) and k (1 ≤ k ≤ 104).

Each of the next n lines contains two integers: ai (1 ≤ ai ≤ h), ci (1 ≤ ci ≤ 109). That means the i-th \"Treasure Cell\" is the ai-th cell and cost of the treasure in that cell is ci dollars. All the ai are distinct.

Each of the next m lines is in one of the three following formats:

  * \"1 x\" — an operation of type 1, 1 ≤ x ≤ h; 
  * \"2 x y\" — an operation of type 2, 1 ≤ x ≤ n, 0 ≤ y < cx; 
  * \"3\" — an operation of type 3. 



There are at most 20 operations of type 1. It's guaranteed that at any moment treasure in each cell has positive value. It's guaranteed that all operations is correct (no operation can decrease the value of the taken tresure).

Please, do not use the %lld specifier to read 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

For each operation of type 3, output an integer indicates the value (in dollars) of the most valuable treasure among the \"Treasure Cells\" Freda can reach. If there is no such treasure, output 0.

Examples

Input

10 3 5 2
5 50
7 60
8 100
2 2 5
3
1 3
3
3


Output

55
100
50

Note

In the sample, there are 10 cells and 3 \"Treasure Cells\". The first \"Treasure Cell\" is cell 5, having 50 dollars tresure in it. The second \"Treasure Cell\" is cell 7, having 60 dollars tresure in it. The third \"Treasure Cell\" is cell 8, having 100 dollars tresure in it.

At first, Freda can only reach cell 1, 3, 5, 7 and 9. In the first operation, we reduce the value in the second \"Treasure Cell\" from 60 to 55. Then the most valuable treasure among the \"Treasure Cells\" she can reach is max(50, 55) = 55. After the third operation, she can also go 3 cells forward each step, being able to reach cell 1, 3, 4, 5, 6, 7, 8, 9, 10. So the most valuable tresure is 100.

Noticed that she took the 55 dollars and 100 dollars treasure away, so the last answer is 50.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
struct record {
  long long x, y;
  bool operator<(const record &b) const { return x == b.x ? y > b.y : x < b.x; }
  record() {}
  record(long long _x, long long _y) {
    x = _x;
    y = _y;
  }
};
vector<long long> p[10010];
long long a[100010];
long long c[100010];
long long d[10010];
priority_queue<record> q;
long long newk[100010];
long long cnt = 0;
long long h, n, m, k;
bool cmp(long long i, long long j) { return a[i] < a[j]; }
void spfa() {
  queue<long long> bfs;
  bool flag[10010];
  for (long long i = 0; i < k; i++)
    if (d[i] < 1e18 + 100) {
      bfs.push(i);
      flag[i] = true;
    } else
      flag[i] = false;
  while (!bfs.empty()) {
    long long now = bfs.front();
    bfs.pop();
    flag[now] = false;
    for (long long i = 0; i < cnt; i++) {
      long long next = d[now] + newk[i];
      long long y = next % k;
      if (next < d[y]) {
        d[y] = next;
        if (!flag[y]) {
          bfs.push(y);
          flag[y] = true;
        }
      }
    }
  }
  for (long long i = 0; i < k; i++)
    while (p[i].size() > 0 && a[p[i][p[i].size() - 1]] >= d[i]) {
      q.push(record(c[p[i][p[i].size() - 1]], p[i][p[i].size() - 1]));
      p[i].pop_back();
    }
}
int main() {
  cin >> h >> n >> m >> k;
  for (long long i = 0; i < n; i++) {
    cin >> a[i] >> c[i];
    a[i]--;
    p[a[i] % k].push_back(i);
  }
  for (long long i = 0; i < k; i++) {
    sort(p[i].begin(), p[i].end(), cmp);
    d[i] = 1e18 + 100;
  }
  d[0] = 0;
  spfa();
  while (m--) {
    long long op;
    cin >> op;
    if (op == 1) {
      cin >> newk[cnt++];
      spfa();
    } else if (op == 2) {
      long long x, y;
      cin >> x >> y;
      x--;
      c[x] -= y;
      if (d[a[x] % k] <= a[x]) q.push(record(c[x], x));
    } else if (op == 3) {
      while (!q.empty() && q.top().x != c[q.top().y]) q.pop();
      if (!q.empty()) {
        cout << q.top().x << endl;
        c[q.top().y] = 0;
        q.pop();
      } else
        cout << 0 << endl;
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import heapq
from collections import deque, defaultdict
import random
from bootcamp import Basebootcamp

class Efetchthetreasurebootcamp(Basebootcamp):
    def __init__(self, h_max=1e18, n_max=1e5, m_max=1e5, k_max=1e4):
        self.h_max = min(int(h_max), 10**18)
        self.n_max = min(int(n_max), 10**5)
        self.m_max = min(int(m_max), 10**5)
        self.k_max = min(int(k_max), 10**4)

    def case_generator(self):
        h = random.randint(10, 100)
        n = random.randint(1, 5)
        m = random.randint(5, 10)
        k = random.randint(1, 5)
        
        # Generate unique treasure positions
        ai_pool = random.sample(range(1, h+1), n)
        treasures = [{
            'ai': ai,
            'ci': random.randint(10, 1000)
        } for ai in ai_pool]

        operations = []
        add_ops = 0
        for _ in range(m):
            if add_ops < 20 and random.random() < 0.2:
                # Type 1 operation
                operations.append({
                    'type': 1,
                    'x': random.randint(1, h)
                })
                add_ops += 1
            elif random.random() < 0.4 and n > 0:
                # Type 2 operation
                x = random.randint(1, n)
                y = random.randint(1, treasures[x-1]['ci']-1)
                operations.append({
                    'type': 2,
                    'x': x,
                    'y': y
                })
            else:
                # Type 3 operation
                operations.append({'type': 3})

        return {
            'h': h,
            'n': n,
            'm': m,
            'k': k,
            'treasures': sorted(treasures, key=lambda t: t['ai']),
            'operations': operations
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = [
            f"{question_case['h']} {question_case['n']} {question_case['m']} {question_case['k']}"
        ]
        for treasure in question_case['treasures']:
            input_lines.append(f"{treasure['ai']} {treasure['ci']}")
        for op in question_case['operations']:
            if op['type'] == 1:
                input_lines.append(f"1 {op['x']}")
            elif op['type'] == 2:
                input_lines.append(f"2 {op['x']} {op['y']}")
            else:
                input_lines.append("3")
        input_str = '\n'.join(input_lines)
        prompt = f"""你是程序员Freda，需要处理一组操作。初始时，Freda站在第1个单元格，可以跳k步。每次操作可以是添加新跳跃方式、减少宝藏价值或查询最大价值宝藏。输出每个类型3操作的结果。

输入格式：
第一行：h n m k
n行：ai ci
m行：操作

请将类型3的输出按顺序用空格分隔，放在[answer]和[/answer]之间。

输入数据：
{input_str}

答案格式：[answer]结果1 结果2 ...[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        return ' '.join(last_match.split())

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # Initialize data structures
            h = identity['h']
            n = identity['n']
            m = identity['m']
            k = identity['k']
            treasures = identity['treasures']
            operations = identity['operations']

            # Preprocess treasures
            ai_list = [t['ai'] for t in treasures]
            ci_list = [t['ci'] for t in treasures]
            removed = [False] * n

            # Initialize reachable cells
            current_methods = [k]
            reachable = set()
            visited = set()
            queue = deque([1])
            
            while queue:
                pos = queue.popleft()
                if pos in visited:
                    continue
                visited.add(pos)
                reachable.add(pos)
                for method in current_methods:
                    next_pos = pos + method
                    if next_pos <= h and next_pos not in visited:
                        queue.append(next_pos)

            # Process operations
            output = []
            for op in operations:
                if op['type'] == 1:
                    # Add new method
                    x = op['x']
                    if x not in current_methods:
                        current_methods.append(x)
                        # Update reachable cells
                        new_reachable = set()
                        for pos in reachable:
                            next_pos = pos + x
                            while next_pos <= h:
                                if next_pos not in reachable:
                                    new_reachable.add(next_pos)
                                next_pos += x
                        reachable.update(new_reachable)
                elif op['type'] == 2:
                    # Reduce treasure value
                    x = op['x'] - 1  # 0-based index
                    y = op['y']
                    ci_list[x] -= y
                elif op['type'] == 3:
                    # Query max value
                    max_val = 0
                    max_idx = -1
                    for i in range(n):
                        if not removed[i] and ai_list[i] in reachable:
                            if ci_list[i] > max_val or (ci_list[i] == max_val and ai_list[i] < ai_list[max_idx]):
                                max_val = ci_list[i]
                                max_idx = i
                    if max_val > 0:
                        output.append(max_val)
                        removed[max_idx] = True
                    else:
                        output.append(0)
            
            # Validate solution
            expected = list(map(str, output))
            actual = solution.split() if solution else []
            return expected == actual
        except:
            return False
