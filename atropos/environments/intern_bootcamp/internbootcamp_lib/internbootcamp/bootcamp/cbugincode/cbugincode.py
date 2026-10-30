"""# 

### 谜题描述
Recently a serious bug has been found in the FOS code. The head of the F company wants to find the culprit and punish him. For that, he set up an organizational meeting, the issue is: who's bugged the code? Each of the n coders on the meeting said: 'I know for sure that either x or y did it!'

The head of the company decided to choose two suspects and invite them to his office. Naturally, he should consider the coders' opinions. That's why the head wants to make such a choice that at least p of n coders agreed with it. A coder agrees with the choice of two suspects if at least one of the two people that he named at the meeting was chosen as a suspect. In how many ways can the head of F choose two suspects?

Note that even if some coder was chosen as a suspect, he can agree with the head's choice if he named the other chosen coder at the meeting.

Input

The first line contains integers n and p (3 ≤ n ≤ 3·105; 0 ≤ p ≤ n) — the number of coders in the F company and the minimum number of agreed people.

Each of the next n lines contains two integers xi, yi (1 ≤ xi, yi ≤ n) — the numbers of coders named by the i-th coder. It is guaranteed that xi ≠ i, yi ≠ i, xi ≠ yi.

Output

Print a single integer –– the number of possible two-suspect sets. Note that the order of the suspects doesn't matter, that is, sets (1, 2) и (2, 1) are considered identical.

Examples

Input

4 2
2 3
1 4
1 4
2 1


Output

6


Input

8 6
5 6
5 7
5 8
6 2
2 1
7 3
1 3
1 4


Output

1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int cnt[300010];
int freq[300010];
void inc(int k) {
  cnt[k]++;
  freq[cnt[k]]++;
}
void dec(int k) {
  freq[cnt[k]]--;
  cnt[k]--;
}
void clear(int k) {
  while (cnt[k] >= 0) {
    freq[cnt[k]]--;
    --cnt[k];
  }
}
pair<int, int> p[300010];
int main() {
  int n, k;
  scanf(\"%d%d\", &n, &k);
  freq[0] = n;
  for (int i = 0; i < n; i++) {
    scanf(\"%d%d\", &p[i].first, &p[i].second);
    if (p[i].first > p[i].second) {
      swap(p[i].first, p[i].second);
    }
    inc(p[i].first);
    inc(p[i].second);
  }
  sort(p, p + n);
  int st = 0, nd = 0, t;
  long long ans = 0;
  for (int i = 1; i <= n; i++) {
    while (nd < n && p[nd].first == i) {
      ++nd;
    }
    for (int t = st; t < nd; t++) {
      dec(p[t].first);
      dec(p[t].second);
    }
    int r = k - (nd - st) - cnt[i];
    clear(i);
    ans += freq[max(r, 0)];
    for (int t = st; t < nd; t++) {
      inc(p[t].second);
    }
    st = nd;
  }
  printf(\"%I64d\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from itertools import combinations

class Cbugincodebootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=10, p_min=0, p_max=None):
        super().__init__()
        self.n_min = max(3, n_min)  # 确保最小值为3
        self.n_max = min(300000, n_max)  # 设置题目上限
        self.p_min = max(0, p_min)
        self.p_max = p_max if p_max is not None else self.n_max

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        p = random.randint(max(self.p_min, 0), min(n, self.p_max))
        
        pairs = []
        for i in range(1, n+1):
            available = list(set(range(1, n+1)) - {i})
            while True:
                x, y = random.sample(available, 2)
                # 确保生成的pair保持原始顺序
                if x < y:  
                    pairs.append([x, y])
                    break
                elif y < x:
                    pairs.append([y, x])
                    break
        return {
            'n': n,
            'p': p,
            'pairs': pairs
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        p = question_case['p']
        pairs = question_case['pairs']
        problem = f"""## Programming Problem

**Background**:
A software bug occurred in a company with {n} coders. Each coder claimed two suspects. The CEO needs to select two suspects such that at least {p} coders approve the choice. A coder approves if at least one of their named suspects is selected.

**Input Format**:
- First line: n p
- Next {n} lines: x y (each coder's claim)

**Current Case**:
{n} {p}
"""
        problem += '\n'.join(f"{x} {y}" for x, y in pairs)
        problem += "\n\n**Output**: The number of valid pairs. Place your final answer within [answer][/answer] tags."
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip().split()[-1])  # 提取最后一个答案的数字部分
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            p = identity['p']
            pairs = identity['pairs']
            
            # 使用字典加速查找
            accusation_map = {i: set(pair) for i, pair in enumerate(pairs, 1)}
            
            valid_pairs = 0
            for u, v in combinations(range(1, n+1), 2):
                count = 0
                for acc in accusation_map.values():
                    if u in acc or v in acc:
                        count += 1
                        if count >= p:  # 提前终止
                            break
                if count >= p:
                    valid_pairs += 1
            return int(solution) == valid_pairs
        except Exception as e:
            print(f"Verification error: {e}")
            return False
