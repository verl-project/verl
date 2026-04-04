"""# 

### 谜题描述
Lesha plays the recently published new version of the legendary game hacknet. In this version character skill mechanism was introduced. Now, each player character has exactly n skills. Each skill is represented by a non-negative integer ai — the current skill level. All skills have the same maximum level A.

Along with the skills, global ranking of all players was added. Players are ranked according to the so-called Force. The Force of a player is the sum of the following values:

  * The number of skills that a character has perfected (i.e., such that ai = A), multiplied by coefficient cf.
  * The minimum skill level among all skills (min ai), multiplied by coefficient cm. 



Now Lesha has m hacknetian currency units, which he is willing to spend. Each currency unit can increase the current level of any skill by 1 (if it's not equal to A yet). Help him spend his money in order to achieve the maximum possible value of the Force.

Input

The first line of the input contains five space-separated integers n, A, cf, cm and m (1 ≤ n ≤ 100 000, 1 ≤ A ≤ 109, 0 ≤ cf, cm ≤ 1000, 0 ≤ m ≤ 1015).

The second line contains exactly n integers ai (0 ≤ ai ≤ A), separated by spaces, — the current levels of skills.

Output

On the first line print the maximum value of the Force that the character can achieve using no more than m currency units.

On the second line print n integers a'i (ai ≤ a'i ≤ A), skill levels which one must achieve in order to reach the specified value of the Force, while using no more than m currency units. Numbers should be separated by spaces.

Examples

Input

3 5 10 1 5
1 3 1


Output

12
2 5 2 


Input

3 5 10 1 339
1 3 1


Output

35
5 5 5 

Note

In the first test the optimal strategy is to increase the second skill to its maximum, and increase the two others by 1.

In the second test one should increase all skills to maximum.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int main() {
  long long N, A, C1, C2, M;
  cin >> N >> A >> C1 >> C2 >> M;
  long long M_store = M;
  vector<pair<int, int> > I(N);
  long long total = 0;
  for (int i = 0; i < N; i++) {
    scanf(\"%d\", &I[i].first);
    total += I[i].first;
    I[i].second = i;
  }
  sort(I.begin(), I.end());
  if (total + M >= A * N) {
    cout << C1 * N + C2 * A << endl;
    for (int i = 0; i < N; i++) {
      printf(\"%I64d\", A);
      printf(i + 1 == N ? \"\n\" : \" \");
    }
    return 0;
  }
  long long result = 0;
  for (int T : {1, 2}) {
    M = M_store;
    long long currVal = 0;
    vector<int> IB(N);
    for (int i = N - 1; i >= 0; i--) {
      int fill = A - I[i].first;
      if (M < fill)
        fill = M;
      else
        currVal += C1;
      M -= fill;
      IB[i] = I[i].first + fill;
    }
    long long next_i = 1;
    long long f_val = I[0].first;
    long long f_cnt = 1;
    for (int i = 0; i <= N; i++) {
      result = max(result, currVal + (f_val / f_cnt) * C2);
      if (T == 2 and result == currVal + (f_val / f_cnt) * C2) {
        for (int j = 0; j < f_cnt; j++) {
          IB[j] = f_val / f_cnt;
          if (f_val % f_cnt != 0) {
            IB[j]++;
            f_val--;
          }
        }
        vector<int> answer(N);
        for (int j = 0; j < N; j++) answer[I[j].second] = IB[j];
        cout << result << endl;
        for (int j = 0; j < N; j++) {
          printf(\"%d\", answer[j]);
          printf(j + 1 == N ? \"\n\" : \" \");
        }
        return 0;
      }
      if (i == N) break;
      if (IB[i] == A) currVal -= C1;
      f_val += IB[i] - I[i].first;
      IB[i] = I[i].first;
      while (next_i <= i and (f_val / f_cnt) >= IB[next_i]) {
        f_cnt++;
        f_val += IB[next_i];
        next_i++;
      }
    }
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re

class Dskillsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'n_min': 3,
            'n_max': 10,
            'A_min': 50,
            'A_max': 500,
            'cf_min': 1,
            'cf_max': 1000,
            'cm_min': 1,
            'cm_max': 1000,
            'm_min': 0,
            'm_max': 10**18
        }
        self.params.update(params)
    
    def case_generator(self):
        params = self.params
        n = random.randint(params['n_min'], params['n_max'])
        A = random.randint(params['A_min'], params['A_max'])
        cf = random.randint(params['cf_min'], params['cf_max'])
        cm = random.randint(params['cm_min'], params['cm_max'])
        m = random.randint(params['m_min'], params['m_max'])
        
        a_initial = [
            random.randint(0, A-1) if random.random() < 0.8 else A
            for _ in range(n)
        ]
        return {
            'n': n,
            'A': A,
            'cf': cf,
            'cm': cm,
            'm': m,
            'a_initial': a_initial
        }
    
    @staticmethod
    def prompt_func(question_case):  # 此处原先缺少正确缩进
        n = question_case['n']
        A = question_case['A']
        cf = question_case['cf']
        cm = question_case['cm']
        m = question_case['m']
        a_str = ' '.join(map(str, question_case['a_initial']))
        prompt = (
            f"你是游戏Hacknet中的一名角色，需要帮助Lesha分配他的技能点以获得最大的战力值（Force）。\n"
            f"你的角色当前有{n}项技能，每项的当前等级为：{a_str}。每项技能的最高等级为{A}。\n"
            f"Lesha拥有{m}个货币单位，每个单位可以提升任一技能1级（不能超过最高等级A）。\n"
            f"战力值的计算方式为：\n"
            f"- 完美技能的数量（即等于最高等级A的技能数）乘以系数{cf}。\n"
            f"- 所有技能中最低等级的技能等级乘以系数{cm}。\n"
            f"请分配货币，使得战力值最大。\n"
            f"输出格式要求：\n"
            f"第一行输出最大战力值，第二行输出各技能的最终等级，用空格分隔。\n"
            f"请将答案放在[answer]标签内，例如：\n"
            f"[answer]\n最大值\n最终等级列表...\n[/answer]"
        )
        return prompt  # 补全函数体并确保缩进
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = [line.strip() for line in last_match.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            max_force = int(lines[0])
            final_levels = list(map(int, lines[1].split()))
            return {'max_force': max_force, 'final_levels': final_levels}
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or 'final_levels' not in solution or 'max_force' not in solution:
            return False
        final_levels = solution['final_levels']
        claimed_force = solution['max_force']
        n = identity['n']
        A = identity['A']
        cf = identity['cf']
        cm = identity['cm']
        m = identity['m']
        a_initial = identity['a_initial']
        
        # 验证长度匹配
        if len(final_levels) != n:
            return False
        
        # 验证等级范围
        if any(af < ai or af > A for af, ai in zip(final_levels, a_initial)):
            return False
        
        # 验证总花费
        total_cost = sum(af - ai for af, ai in zip(final_levels, a_initial))
        if total_cost > m:
            return False
        
        # 计算实际战力值
        perfect_count = sum(1 for af in final_levels if af == A)
        min_level = min(final_levels)
        actual_force = perfect_count * cf + min_level * cm
        
        return actual_force == claimed_force
    
    @staticmethod
    def compute_max_force(n, A, cf, cm, m, a_initial):
        sorted_a = sorted(a_initial)
        total = sum(sorted_a)
        
        # 处理全满的特殊情况
        if total + m >= n * A:
            return cf * n + cm * A, [A]*n
        
        # 计算可能的最大完美技能数
        perfect = 0
        for i in reversed(range(n)):
            cost = A - sorted_a[i]
            if m >= cost:
                perfect += 1
                m -= cost
            else:
                break
        
        # 提高最低技能
        min_level = sorted_a[0]
        for i in range(1, n-perfect):
            delta = sorted_a[i] - sorted_a[i-1]
            if m >= delta * i:
                min_level += delta
                m -= delta * i
            else:
                min_level += m // i
                m %= i
                break
        
        final_force = perfect * cf + min_level * cm
        final_levels = [max(a, min_level) for a in a_initial]
        # 升满完美技能
        for i in reversed(range(n)):
            if final_levels[i] < A and perfect > 0:
                final_levels[i] = A
                perfect -= 1
        return final_force, final_levels
