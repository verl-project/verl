"""# 

### 谜题描述
You need to execute several tasks, each associated with number of processors it needs, and the compute power it will consume.

You have sufficient number of analog computers, each with enough processors for any task. Each computer can execute up to one task at a time, and no more than two tasks total. The first task can be any, the second task on each computer must use strictly less power than the first. You will assign between 1 and 2 tasks to each computer. You will then first execute the first task on each computer, wait for all of them to complete, and then execute the second task on each computer that has two tasks assigned.

If the average compute power per utilized processor (the sum of all consumed powers for all tasks presently running divided by the number of utilized processors) across all computers exceeds some unknown threshold during the execution of the first tasks, the entire system will blow up. There is no restriction on the second tasks execution. Find the lowest threshold for which it is possible.

Due to the specifics of the task, you need to print the answer multiplied by 1000 and rounded up.

Input

The first line contains a single integer n (1 ≤ n ≤ 50) — the number of tasks.

The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 108), where ai represents the amount of power required for the i-th task.

The third line contains n integers b1, b2, ..., bn (1 ≤ bi ≤ 100), where bi is the number of processors that i-th task will utilize.

Output

Print a single integer value — the lowest threshold for which it is possible to assign all tasks in such a way that the system will not blow up after the first round of computation, multiplied by 1000 and rounded up.

Examples

Input

6
8 10 9 9 8 10
1 1 1 1 1 1


Output

9000


Input

6
8 10 9 9 8 10
1 10 5 5 1 10


Output

1160

Note

In the first example the best strategy is to run each task on a separate computer, getting average compute per processor during the first round equal to 9.

In the second task it is best to run tasks with compute 10 and 9 on one computer, tasks with compute 10 and 8 on another, and tasks with compute 9 and 8 on the last, averaging (10 + 10 + 9) / (10 + 10 + 5) = 1.16 compute power per processor during the first round.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int ms = 100 + 10;
long long dp[ms][ms], sum[ms];
pair<long long, long long> a[ms], b[ms];
int n, re[ms];
bool cmp(const pair<long long, long long>& l,
         const pair<long long, long long>& r) {
  return l.first < r.first || (l.first == r.first && l.second > r.second);
}
bool check(long long x) {
  memset(dp, 0, sizeof(dp));
  for (int i = 0; i < n; ++i) {
    sum[i] = a[i].first - x * a[i].second;
  }
  sum[n] = 0;
  for (int i = n - 1; i >= 0; --i) sum[i] += sum[i + 1];
  for (int i = n - 1; i >= 0; --i) {
    int r = re[i], k = (r - i + 1);
    for (int j = 0; j < n; ++j) dp[i][j] = dp[i + 1][j];
    for (int j = 0; 2 * j + k < n - r; ++j) {
      dp[i][j + k] = max(dp[i][j + k], dp[r + 1][j] + sum[i] - sum[r + 1]);
    }
  }
  return sum[0] - *max_element(dp[0], dp[0] + n) <= 0;
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cin >> n;
  for (int i = 0; i < n; ++i) cin >> a[i].first, a[i].first *= 1000;
  for (int i = 0; i < n; ++i) cin >> a[i].second;
  sort(a, a + n, cmp);
  for (int i = n - 1; i >= 0; --i) {
    if (a[i].first == a[i + 1].first)
      re[i] = re[i + 1];
    else
      re[i] = i;
  }
  long long l = 0, r = a[n - 1].first, mid, res = 0;
  while (l <= r) {
    mid = (l + r) / 2;
    if (check(mid)) {
      res = mid;
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }
  cout << res;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from functools import cmp_to_key
from bootcamp import Basebootcamp

def compare_pairs(l, r):
    if l[0] < r[0]:
        return -1
    elif l[0] > r[0]:
        return 1
    else:
        return -1 if l[1] > r[1] else 1

def compute_min_threshold(n, a_list, b_list):
    a = [(ai * 1000, bi) for ai, bi in zip(a_list, b_list)]
    a_sorted = sorted(a, key=cmp_to_key(compare_pairs))
    
    re = [0] * n
    re[-1] = n - 1
    for i in range(n-2, -1, -1):
        if a_sorted[i][0] == a_sorted[i+1][0]:
            re[i] = re[i+1]
        else:
            re[i] = i
    
    left = 0
    right = a_sorted[-1][0]
    res = right
    
    while left <= right:
        mid = (left + right) // 2
        sum_list = [0] * (n + 1)
        for i in range(n):
            sum_list[i] = a_sorted[i][0] - mid * a_sorted[i][1]
        sum_list[n] = 0
        for i in range(n-1, -1, -1):
            sum_list[i] += sum_list[i+1]
        
        dp = [[0]*(n+1) for _ in range(n+2)]
        for i in range(n-1, -1, -1):
            r = re[i]
            k = r - i + 1
            for j in range(n+1):
                dp[i][j] = dp[i+1][j]
            max_j = (n - r - k -1) // 2  # 修正循环边界条件
            for j in range(0, max_j + 1):
                if j + k > n:
                    continue
                current_val = dp[r+1][j] + (sum_list[i] - sum_list[r+1])
                if current_val > dp[i][j + k]:
                    dp[i][j + k] = current_val
        
        max_val = max(dp[0])
        if sum_list[0] - max_val <= 0:
            res = mid
            right = mid - 1
        else:
            left = mid + 1
    
    return res

class Fcomputepowerbootcamp(Basebootcamp):
    def __init__(self, max_n=50, max_a=1e8, max_b=100):
        self.max_n = max_n
        self.max_a = int(max_a)
        self.max_b = max_b
    
    def case_generator(self):
        while True:
            n = random.randint(1, self.max_n)
            a = [random.randint(1, self.max_a) for _ in range(n)]
            b = [random.randint(1, self.max_b) for _ in range(n)]
            expected = compute_min_threshold(n, a, b)
            # 确保生成的案例有有效解
            if expected <= a[-1]*1000:  # 阈值不超过最大任务功耗
                return {
                    "n": n,
                    "a": a,
                    "b": b,
                    "expected": expected
                }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        a = question_case['a']
        b = question_case['b']
        n = question_case['n']
        problem_text = f"""您需要配置模拟计算机执行计算任务。每个任务需要特定数量的处理器并消耗算力。

# 规则
1. 每个计算机可运行1或2个任务
2. 若计算机运行2个任务，第二任务算力 **必须严格小于** 第一任务
3. 在首轮执行中，所有计算机的第一个任务必须满足:
   - 算力均值 = (所有首任务算力总和) / (所有首任务处理器总数) ≤ 阈值
4. 找到 **最小可能阈值** ，将阈值 ×1000后 **向上取整** 输出

# 任务参数
任务数量: {n}
各任务算力需求: {a}
各任务处理器需求: {b}

请将最终答案置于[answer]标签内，如[answer]9000[/answer]。"""
        return problem_text
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.I)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
