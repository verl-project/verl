"""# 

### 谜题描述
In the year 2500 the annual graduation ceremony in the German University in Cairo (GUC) has run smoothly for almost 500 years so far.

The most important part of the ceremony is related to the arrangement of the professors in the ceremonial hall.

Traditionally GUC has n professors. Each professor has his seniority level. All seniorities are different. Let's enumerate the professors from 1 to n, with 1 being the most senior professor and n being the most junior professor.

The ceremonial hall has n seats, one seat for each professor. Some places in this hall are meant for more senior professors than the others. More specifically, m pairs of seats are in \"senior-junior\" relation, and the tradition requires that for all m pairs of seats (ai, bi) the professor seated in \"senior\" position ai should be more senior than the professor seated in \"junior\" position bi.

GUC is very strict about its traditions, which have been carefully observed starting from year 2001. The tradition requires that: 

  * The seating of the professors changes every year. 
  * Year 2001 ceremony was using lexicographically first arrangement of professors in the ceremonial hall. 
  * Each consecutive year lexicographically next arrangement of the professors is used. 



The arrangement of the professors is the list of n integers, where the first integer is the seniority of the professor seated in position number one, the second integer is the seniority of the professor seated in position number two, etc.

Given n, the number of professors, y, the current year and m pairs of restrictions, output the arrangement of the professors for this year.

Input

The first line contains three integers n, y and m (1 ≤ n ≤ 16, 2001 ≤ y ≤ 1018, 0 ≤ m ≤ 100) — the number of professors, the year for which the arrangement should be computed, and the number of pairs of seats for which the seniority relation should be kept, respectively.

The next m lines contain one pair of integers each, \"ai bi\", indicating that professor on the ai-th seat is more senior than professor on the bi-th seat (1 ≤ ai, bi ≤ n, ai ≠ bi). Some pair may be listed more than once.

Please, do not use the %lld specificator to read or write 64-bit integers in С++. It is preferred to use the cin stream (you may also use the %I64d specificator).

Output

Print the order in which the professors should be seated in the requested year.

If by this year the GUC would have ran out of arrangements, or the given \"senior-junior\" relation are contradictory, print \"The times have changed\" (without quotes).

Examples

Input

3 2001 2
1 2
2 3


Output

1 2 3


Input

7 2020 6
1 2
1 3
2 4
2 5
3 6
3 7


Output

1 2 3 7 4 6 5


Input

10 3630801 0


Output

The times have changed


Input

3 2001 3
1 2
2 3
3 1


Output

The times have changed

Note

In the first example the lexicographically first order of seating is 1 2 3.

In the third example the GUC will run out of arrangements after the year 3630800.

In the fourth example there are no valid arrangements for the seating.

The lexicographical comparison of arrangements is performed by the < operator in modern programming languages. The arrangement a is lexicographically less that the arrangement b, if there exists such i (1 ≤ i ≤ n), that ai < bi, and for any j (1 ≤ j < i) aj = bj.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n, m, ls[20], pref[20];
long long y, dp[100000];
long long count() {
  fill(dp, dp + (1 << n), 0);
  dp[0] = 1;
  for (int mask = 0; mask < (1 << n); ++mask) {
    if (dp[mask] == 0) continue;
    int cnt = 0, tmp = mask;
    while (tmp > 0) {
      if (tmp & 1 == 1) cnt++;
      tmp /= 2;
    }
    for (int i = 0; i < n; i++)
      if ((pref[i] == -1 || pref[i] == n - cnt - 1) &&
          ((ls[i] & mask) == ls[i]) && ((mask & (1 << i)) == 0))
        dp[mask | (1 << i)] += dp[mask];
  }
  return dp[(1 << n) - 1];
}
int main() {
  cin >> n >> y >> m;
  y -= 2000;
  for (int i = 0; i < m; i++) {
    int u, v;
    cin >> u >> v;
    ls[u - 1] |= 1 << (v - 1);
  }
  fill(pref, pref + n, -1);
  for (int i = 0; i < n; i++) {
    for (;;) {
      pref[i]++;
      if (pref[i] == n) {
        cout << \"The times have changed\" << endl;
        return 0;
      }
      long long tmp = count();
      if (tmp < y)
        y -= tmp;
      else
        break;
    }
  }
  for (int i = 0; i < n; i++) cout << pref[i] + 1 << \" \";
  cout << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque
from bootcamp import Basebootcamp

def is_dag(edges, n_nodes):
    adj = [[] for _ in range(n_nodes + 1)]
    in_degree = [0] * (n_nodes + 1)
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
    queue = deque()
    for node in range(1, n_nodes + 1):
        if in_degree[node] == 0:
            queue.append(node)
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return visited == n_nodes

def calculate_count(n, ls, pref):
    dp = [0] * (1 << n)
    dp[0] = 1
    for mask in range(1 << n):
        if dp[mask] == 0:
            continue
        cnt = bin(mask).count('1')
        for i in range(n):
            if pref[i] != -1 and pref[i] != (n - cnt - 1):
                continue
            if (ls[i] & mask) != ls[i]:
                continue
            if (mask & (1 << i)) != 0:
                continue
            new_mask = mask | (1 << i)
            dp[new_mask] += dp[mask]
    return dp[(1 << n) - 1]

def solve_puzzle(n, y, m, constraints):
    original_y = y
    y -= 2000
    if y <= 0:
        return "The times have changed"
    ls = [0] * n
    for u, v in constraints:
        ai = u - 1
        bi_seat = v - 1
        ls[ai] |= 1 << bi_seat
    pref = [-1] * n
    for i in range(n):
        while True:
            pref[i] += 1
            if pref[i] >= n:
                return "The times have changed"
            current_pref = pref[:i+1] + [-1] * (n - i - 1)
            current_count = calculate_count(n, ls, current_pref)
            if current_count < y:
                y -= current_count
            else:
                break
    arrangement = [str(p + 1) for p in pref]
    return ' '.join(arrangement)

class Earrangementbootcamp(Basebootcamp):
    def __init__(self, max_n=16, **params):
        self.max_n = max_n
        super().__init__(**params)
    
    def case_generator(self):
        # 生成教授数n，确保不超过题目要求的16
        n = random.randint(1, min(16, self.max_n))
        seats = list(range(1, n + 1))
        # 生成m的取值范围：0到min(100, n*(n-1))
        max_possible_m = n * (n - 1)
        m_max = min(100, max_possible_m)
        m = random.randint(0, m_max)
        edges = []
        # 生成m对约束，允许重复和无效约束
        for _ in range(m):
            ai, bi = random.choices(seats, k=2)
            if ai != bi:
                edges.append((ai, bi))
        # 去重（仅提高生成效率，但保留重复的可能性）
        edges = list(set(edges))  # 根据题目要求，重复约束不影响结果，但减少无效尝试
        m = len(edges)
        # 检查约束是否形成DAG
        valid_dag = is_dag(edges, n)
        # 确定y的取值范围
        if not valid_dag:
            # 约束无效时，任何年份都应返回错误
            y = random.randint(2001, 2001 + 100)
        else:
            try:
                ls = [0] * n
                for u, v in edges:
                    ai = u - 1
                    bi_seat = v - 1
                    ls[ai] |= 1 << bi_seat
                pref = [-1] * n
                total = calculate_count(n, ls, pref)
                if total == 0:
                    valid_dag = False  # 虽然约束是DAG，但无解
                else:
                    # 50%概率生成有效年份，50%生成超限年份
                    if random.random() < 0.5:
                        y = 2000 + random.randint(1, total)
                    else:
                        y = 2000 + total + random.randint(1, 3)
            except:
                valid_dag = False
        # 最终生成案例
        return {
            'n': n,
            'y': y,
            'm': m,
            'constraints': edges
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        y = question_case['y']
        m = question_case['m']
        constraints = question_case['constraints']
        problem = (
            f"In the year 2500, the GUC graduation ceremony requires professors to be seated with specific seniority rules.\n"
            f"There are {n} professors (seniority 1 to {n}) and {m} seat relations.\n"
            f"Determine the seating arrangement for year {y} considering lexicographical order.\n"
            f"Constraints:\n" + 
            '\n'.join(f"{ai} {bi}" for ai, bi in constraints) + 
            "\n\nOutput the arrangement as space-separated numbers or state 'The times have changed' within [answer] tags. "
            "Example:\n[answer]1 2 3[/answer]\nOR\n[answer]The times have changed[/answer]"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        # 使用正则表达式匹配最后一个[answer]标签中的内容
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        # 处理可能的换行和多余空格
        last_answer = re.sub(r'\s+', ' ', last_answer).strip()
        return last_answer
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        # 标准化solution的格式
        solution = solution.strip().lower()
        # 获取正确解
        try:
            correct = solve_puzzle(
                identity['n'],
                identity['y'],
                identity['m'],
                identity['constraints']
            )
            # 处理两种可能的结果
            if correct == "The times have changed":
                return solution == "the times have changed"
            else:
                # 将答案转换为统一格式（如去除多余空格）
                correct_clean = re.sub(r'\s+', ' ', correct).strip()
                solution_clean = re.sub(r'\s+', ' ', solution).strip()
                return correct_clean == solution_clean
        except:
            return solution == "the times have changed"
