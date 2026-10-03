"""# 

### 谜题描述
You are running for a governor in a small city in Russia. You ran some polls and did some research, and for every person in the city you know whom he will vote for, and how much it will cost to bribe that person to vote for you instead of whomever he wants to vote for right now. You are curious, what is the smallest amount of money you need to spend on bribing to win the elections. To win elections you need to have strictly more votes than any other candidate.

Input

First line contains one integer n (1 ≤ n ≤ 105) — number of voters in the city. Each of the next n lines describes one voter and contains two integers ai and bi (0 ≤ ai ≤ 105; 0 ≤ bi ≤ 104) — number of the candidate that voter is going to vote for and amount of money you need to pay him to change his mind. You are the candidate 0 (so if a voter wants to vote for you, ai is equal to zero, in which case bi will also be equal to zero).

Output

Print one integer — smallest amount of money you need to spend to win the elections.

Examples

Input

5
1 2
1 2
1 2
2 1
0 0


Output

3


Input

4
1 2
1 2
2 1
0 0


Output

2


Input

1
100000 0


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 100100;
const int MAXP = (1 << 18) + 100;
const int P = (1 << 17);
int N;
int a[MAXN], b[MAXN];
vector<int> v[MAXN];
vector<int> add[MAXN];
int seg[MAXP];
int ucnt[MAXP];
void upd(int cloc, int x, int y) {
  cloc += P;
  while (cloc > 0) {
    seg[cloc] += x;
    ucnt[cloc] += y;
    cloc /= 2;
  }
}
int usum(int cloc) {
  cloc += P;
  int res = 0;
  while (cloc > 1) {
    if (cloc % 2 == 1) res += ucnt[cloc - 1];
    cloc /= 2;
  }
  return res;
}
int ssum(int cloc) {
  cloc += P;
  int res = 0;
  while (cloc > 1) {
    if (cloc % 2 == 1) res += seg[cloc - 1];
    cloc /= 2;
  }
  return res;
}
int floc(int x) {
  if (usum(N) < x) return 1e9;
  int lo = 0, hi = N;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (usum(mid) >= x)
      hi = mid;
    else
      lo = mid + 1;
  }
  return ssum(lo);
}
int ord[MAXN];
int fnext(int x) {
  int lo = 0, hi = N - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (ord[mid] < x || (ord[mid] == x && ucnt[mid + P] == 1))
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}
int main() {
  cin >> N;
  for (int i = 0; i < N; i++) {
    cin >> a[i] >> b[i];
    v[a[i]].push_back(b[i]);
    ord[i] = b[i];
  }
  sort(ord, ord + N);
  int ctot = N;
  int csum = 0;
  for (int i = 1; i < MAXN; i++) {
    sort(v[i].begin(), v[i].end());
    reverse(v[i].begin(), v[i].end());
    for (int j = 0; j < v[i].size(); j++) {
      csum += v[i][j];
      add[j].push_back(v[i][j]);
    }
  }
  int ans = 2e9;
  for (int i = 0; i < MAXN; i++) {
    int need = i + 1;
    ans = min(ans, csum + floc(need - ctot));
    for (int j = 0; j < add[i].size(); j++) {
      ctot--;
      csum -= add[i][j];
      upd(fnext(add[i][j]), add[i][j], 1);
    }
  }
  cout << ans << \"\n\";
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Celectionsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_voters = params.get('max_voters', 20)
        self.max_bribe = params.get('max_bribe', 100)
        self.max_candidates = params.get('max_candidates', 3)
    
    def case_generator(self):
        for _ in range(100):  # 重试次数上限
            n = random.randint(1, self.max_voters)
            my_votes = random.randint(0, n)
            remaining = n - my_votes

            # 处理全票支持自己的情况
            if remaining == 0:
                return {
                    'n': n,
                    'voters': [(0,0)] * n,
                    'min_cost': 0
                }

            # 修正候选人数量生成逻辑
            max_possible_candidates = min(self.max_candidates, remaining)
            others_num = random.randint(1, max_possible_candidates)
            
            # 确保每个候选人至少获得1票
            base_counts = [1] * others_num
            remaining_after_base = remaining - others_num
            if remaining_after_base < 0:
                continue  # 无法满足最小分配条件，重新生成

            # 分配剩余票数
            for _ in range(remaining_after_base):
                idx = random.randint(0, others_num-1)
                base_counts[idx] += 1

            # 构建选民数据
            voters = [(0,0) for _ in range(my_votes)]
            for i in range(others_num):
                candidate = i + 1
                count = base_counts[i]
                # 生成贿赂成本并排序确保贪心算法有效性
                bribes = sorted([random.randint(0, self.max_bribe) for _ in range(count)], reverse=True)
                voters.extend([(candidate, b) for b in bribes])

            random.shuffle(voters)  # 随机打乱顺序

            # 计算最小成本
            min_cost = self.calculate_min_cost(voters)
            if min_cost is not None and min_cost != float('inf'):
                return {
                    'n': n,
                    'voters': voters,
                    'min_cost': min_cost
                }
        raise ValueError("无法生成有效案例，请调整参数设置")

    @staticmethod
    def calculate_min_cost(voters):
        c0 = sum(1 for ai, _ in voters if ai == 0)
        candidate_bribes = defaultdict(list)
        
        # 收集贿赂成本并按候选人分组
        for ai, bi in voters:
            if ai != 0:
                candidate_bribes[ai].append(bi)
        
        # 对每个候选人的贿赂成本排序（降序，便于后续处理）
        for k in candidate_bribes:
            candidate_bribes[k].sort(reverse=True)
        
        # 预处理所有可能的贿赂方案
        all_costs = []
        total_available = 0
        for cand in candidate_bribes.values():
            all_costs.extend(cand)
            total_available += len(cand)
        
        # 处理无需贿赂的情况
        if not candidate_bribes:
            return 0
        
        # 预处理每个候选人的前缀和
        prefix_sums = {}
        for cand, costs in candidate_bribes.items():
            prefix = [0]
            s = 0
            for cost in costs:
                s += cost
                prefix.append(s)
            prefix_sums[cand] = prefix
        
        min_cost = float('inf')
        max_possible = c0 + total_available
        
        # 确定s的范围优化：s只需要到达最大候选人的当前票数+1
        max_current_votes = max(len(v) for v in candidate_bribes.values())
        s_candidates = range(max(1, max_current_votes - c0 + 1), max_possible + 1)
        if not s_candidates:
            return float('inf')
        
        # 计算所有可能的s值
        for s in s_candidates:
            required = s - c0
            if required <= 0:
                current_cost = 0
                if all(len(v) < s for v in candidate_bribes.values()):
                    current_cost = 0
                else:
                    continue
            else:
                total_bribes = 0
                total_obtained = 0
                remaining_costs = []
                
                # 第一部分：必须贿赂的选票
                for cand, costs in candidate_bribes.items():
                    needed = max(len(costs) - (s - 1), 0)
                    if needed > len(costs):
                        break
                    total_bribes += prefix_sums[cand][needed]
                    total_obtained += needed
                    remaining_costs.extend(costs[needed:])
                else:  # 正常完成循环时才执行后续逻辑
                    # 第二部分：补充需要的额外选票
                    if total_obtained >= required:
                        current_cost = total_bribes
                    else:
                        additional_needed = required - total_obtained
                        if len(remaining_costs) < additional_needed:
                            continue
                        remaining_sorted = sorted(remaining_costs)
                        current_cost = total_bribes + sum(remaining_sorted[:additional_needed])
                    
                    if current_cost < min_cost:
                        min_cost = current_cost
        
        return min_cost if min_cost != float('inf') else None

    @staticmethod
    def prompt_func(question_case) -> str:
        voters = question_case['voters']
        n = question_case['n']
        input_lines = [f"{n}"]
        for ai, bi in voters:
            input_lines.append(f"{ai} {bi}")
        input_str = "\n".join(input_lines)
        prompt = (
            "你正在参与俄罗斯一个小城市的市长选举。你需要计算确保你的得票数严格超过其他所有候选人的最小贿赂金额。每个选民用两个整数表示：ai表示当前支持的候选人编号（0表示你），bi表示让该选民改投你所需的金额。\n\n"
            "输入格式：\n"
            "第一行包含整数n（选民总数）。随后n行每行两个整数ai和bi。\n\n"
            "当前问题：\n"
            f"{input_str}\n\n"
            "输出要求：\n"
            "输出一个整数表示最小花费，用[answer]和[/answer]标签包裹答案。例如：[answer]42[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['min_cost']
