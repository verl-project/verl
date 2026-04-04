"""# 

### 谜题描述
The Little Elephant loves the LCM (least common multiple) operation of a non-empty set of positive integers. The result of the LCM operation of k positive integers x1, x2, ..., xk is the minimum positive integer that is divisible by each of numbers xi.

Let's assume that there is a sequence of integers b1, b2, ..., bn. Let's denote their LCMs as lcm(b1, b2, ..., bn) and the maximum of them as max(b1, b2, ..., bn). The Little Elephant considers a sequence b good, if lcm(b1, b2, ..., bn) = max(b1, b2, ..., bn).

The Little Elephant has a sequence of integers a1, a2, ..., an. Help him find the number of good sequences of integers b1, b2, ..., bn, such that for all i (1 ≤ i ≤ n) the following condition fulfills: 1 ≤ bi ≤ ai. As the answer can be rather large, print the remainder from dividing it by 1000000007 (109 + 7).

Input

The first line contains a single positive integer n (1 ≤ n ≤ 105) — the number of integers in the sequence a. The second line contains n space-separated integers a1, a2, ..., an (1 ≤ ai ≤ 105) — sequence a.

Output

In the single line print a single integer — the answer to the problem modulo 1000000007 (109 + 7).

Examples

Input

4
1 4 3 2


Output

15


Input

2
6 3


Output

13

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int max_n = 100000;
const int max_val = 100000;
const int mod = 1000000007;
int mod_val(int a) {
  a %= mod;
  if (a < 0) a += mod;
  return a;
}
int times_mod(int a, int b) {
  long long int x = a;
  x *= b;
  x %= mod;
  return mod_val(x);
}
int power_mod(int a, int b) {
  if (b == 0)
    return 1;
  else if (b == 1)
    return mod_val(a);
  int tmp = power_mod(a, b / 2);
  tmp = times_mod(tmp, tmp);
  if (b % 2)
    return times_mod(tmp, a);
  else
    return tmp;
}
int n;
int a[max_n];
void get_input() {
  scanf(\"%d\", &n);
  for (int i = 0; i < n; i++) scanf(\"%d\", &a[i]);
}
int dist[max_val + 1];
vector<int> divisor[max_val + 1];
void make_dist() {
  for (int i = 0; i <= max_val; i++) dist[i] = 0;
  for (int i = 0; i < n; i++) dist[a[i]]++;
  for (int i = max_val - 1; i >= 0; i--) dist[i] += dist[i + 1];
}
void make_divisor() {
  for (int i = 0; i <= max_val; i++) divisor[i].clear();
  for (int d = 1; d <= max_val; d++)
    for (int v = d; v <= max_val; v += d) divisor[v].push_back(d);
}
int ans;
void process() {
  make_dist();
  make_divisor();
  ans = 1;
  for (int X = 2; X <= max_val; X++) {
    int big = 1;
    int sz = divisor[X].size();
    for (int j = 0; j < sz - 1; j++)
      big = times_mod(
          big, power_mod(j + 1, dist[divisor[X][j]] - dist[divisor[X][j + 1]]));
    big = times_mod(big, power_mod(sz, dist[divisor[X][sz - 1]]));
    int small = 1;
    for (int j = 0; j < sz - 2; j++)
      small = times_mod(small, power_mod(j + 1, dist[divisor[X][j]] -
                                                    dist[divisor[X][j + 1]]));
    small = times_mod(small, power_mod(sz - 1, dist[divisor[X][sz - 2]]));
    ans = mod_val(ans + mod_val(big - small));
  }
}
void print_output() { printf(\"%d\n\", ans); }
int main() {
  get_input();
  process();
  print_output();
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Elittleelephantandlcmbootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_a=10):
        self.max_n = max_n  # 控制生成序列的最大长度
        self.max_a = max_a  # 控制生成元素的最大值
    
    def case_generator(self):
        # 生成有效测试用例并预计算正确答案
        n = random.randint(1, self.max_n)
        a = [random.randint(1, self.max_a) for _ in range(n)]
        
        # 确保至少包含一个1的测试用例
        if random.random() < 0.3:
            a[random.randint(0, n-1)] = 1
            
        correct_answer = self._solve(a)
        return {
            "n": n,
            "a": a,
            "correct_answer": correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        # 构造详细的问题描述
        n = question_case['n']
        a = question_case['a']
        problem_text = f"""你是数学问题解决专家，请解决以下模运算组合计数问题：

给定长度n={n}的整数序列a: {' '.join(map(str, a))}
找出满足以下条件的整数序列b的数量（模1,000,000,007）：
1. 每个b_i满足1 ≤ b_i ≤ a_i
2. b序列的LCM等于其最大值

输入格式：n
           a1 a2 ... an
输出格式：输出一个整数

请将最终答案放在[answer]标签内，例如：[answer]123[/answer]"""
        return problem_text
    
    @staticmethod
    def extract_output(output):
        # 强化答案抽取逻辑
        matches = re.findall(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output)
        if matches:
            try:
                return int(matches[-1]) % MOD
            except ValueError:
                pass
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 严格验证答案的正确性
        return solution == identity['correct_answer']
    
    @staticmethod
    def _solve(a):
        # 优化后的高效解法实现
        if not a:
            return 0
        
        # 预处理频率统计
        freq = defaultdict(int)
        max_val = max(a) if a else 0
        for num in a:
            freq[num] += 1

        # 构建dist数组
        dist = {}
        current = 0
        for x in range(max_val, 0, -1):
            current += freq.get(x, 0)
            dist[x] = current

        # 预计算所有数的约数
        divisors = defaultdict(list)
        for d in range(1, max_val + 1):
            for multiple in range(d, max_val + 1, d):
                divisors[multiple].append(d)

        ans = 1  # 初始值对应X=1的情况

        # 主计算逻辑
        for X in range(2, max_val + 1):
            divs = divisors.get(X, [])
            sz = len(divs)
            if sz < 1:
                continue

            # 计算big乘积项
            big = 1
            for j in range(sz - 1):
                d_current = divs[j]
                d_next = divs[j+1]
                cnt = dist.get(d_current, 0) - dist.get(d_next, 0)
                big = (big * pow(j+1, cnt, MOD)) % MOD
            
            # 处理最后一个约数项
            last_d = divs[-1]
            big = (big * pow(sz, dist.get(last_d, 0), MOD)) % MOD

            # 计算small乘积项
            small = 1
            if sz >= 2:
                for j in range(sz - 2):
                    d_current = divs[j]
                    d_next = divs[j+1]
                    cnt = dist.get(d_current, 0) - dist.get(d_next, 0)
                    small = (small * pow(j+1, cnt, MOD)) % MOD
                
                second_last_d = divs[-2]
                small = (small * pow(sz-1, dist.get(second_last_d, 0), MOD)) % MOD
            else:
                small = 0

            # 累加有效贡献
            contribution = (big - small) % MOD
            ans = (ans + contribution) % MOD

        return ans % MOD
