"""# 

### 谜题描述
Little Petya likes positive integers a lot. Recently his mom has presented him a positive integer a. There's only one thing Petya likes more than numbers: playing with little Masha. It turned out that Masha already has a positive integer b. Petya decided to turn his number a into the number b consecutively performing the operations of the following two types:

  1. Subtract 1 from his number. 
  2. Choose any integer x from 2 to k, inclusive. Then subtract number (a mod x) from his number a. Operation a mod x means taking the remainder from division of number a by number x. 



Petya performs one operation per second. Each time he chooses an operation to perform during the current move, no matter what kind of operations he has performed by that moment. In particular, this implies that he can perform the same operation any number of times in a row.

Now he wonders in what minimum number of seconds he could transform his number a into number b. Please note that numbers x in the operations of the second type are selected anew each time, independently of each other.

Input

The only line contains three integers a, b (1 ≤ b ≤ a ≤ 1018) and k (2 ≤ k ≤ 15).

Please do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

Print a single integer — the required minimum number of seconds needed to transform number a into number b.

Examples

Input

10 1 4


Output

6


Input

6 3 10


Output

2


Input

1000000000000000000 1 3


Output

666666666666666667

Note

In the first sample the sequence of numbers that Petya gets as he tries to obtain number b is as follows: 10  →  8  →  6  →  4  →  3  →  2  →  1.

In the second sample one of the possible sequences is as follows: 6  →  4  →  3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long a, b;
int k;
int mod;
inline int gcd(int a, int b) { return b ? gcd(b, a % b) : a; }
const int maxn = 400000;
int d[maxn];
int get(int r1, int r2) {
  assert(r1 <= r2);
  d[r1] = 0;
  int r[k];
  for (int i = 0; i < k; i++) r[i] = r1 % (i + 1);
  for (int i = r1 + 1; i <= r2; i++) {
    for (int j = 0; j < k; j++)
      if (++r[j] == j + 1) r[j] = 0;
    d[i] = d[i - 1] + 1;
    for (int j = 0; j < k; j++) {
      if (r[j] && i - r[j] >= r1) d[i] = min(d[i], d[i - r[j]] + 1);
    }
  }
  return d[r2];
}
long long solve() {
  int ra = a % mod, rb = b % mod;
  if (a - b < mod && ra >= rb) return get(rb, ra);
  long long ans = (get(rb, mod - 1) + 1) + get(0, ra);
  ans += (get(0, mod - 1) + 1) * (((a - ra) - (b - rb + mod)) / mod);
  return ans;
}
int main() {
  while (scanf(\"%lld\"
               \"%lld\"
               \"%d\",
               &a, &b, &k) >= 1) {
    mod = 1;
    for (int i = 2; i <= k; i++) mod = mod * i / gcd(i, mod);
    assert(maxn > mod);
    printf(
        \"%lld\"
        \"\n\",
        solve());
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import math
from bootcamp import Basebootcamp

def lcm(a, b):
    return a * b // math.gcd(a, b)

def compute_mod(k):
    mod = 1
    for x in range(2, k+1):
        mod = lcm(mod, x)
    return mod

def dynamic_get(r1, r2, k):
    x_list = list(range(2, k+1))
    max_r = r2
    d = [float('inf')] * (max_r + 1)
    d[r1] = 0

    current_mods = [r1 % x for x in x_list]

    for i in range(r1 + 1, r2 + 1):
        new_mods = []
        min_steps = d[i-1] + 1
        for idx, x in enumerate(x_list):
            new_mod = current_mods[idx] + 1
            if new_mod >= x:
                new_mod = 0
            new_mods.append(new_mod)
            if new_mod != 0 and i - new_mod >= r1:
                candidate = d[i - new_mod] + 1
                if candidate < min_steps:
                    min_steps = candidate
        current_mods = new_mods
        d[i] = min_steps
    return d[r2]

def solve(a, b, k):
    if a == b:
        return 0
    mod = compute_mod(k)
    ra = a % mod
    rb = b % mod

    if a - b < mod and ra >= rb:
        return dynamic_get(rb, ra, k)
    else:
        part1 = dynamic_get(rb, mod - 1, k) + 1  # 上升到模的倍数
        part2 = dynamic_get(0, ra, k)
        cycle_num = (a - ra - (b - rb + mod)) // mod
        part3 = (dynamic_get(0, mod-1, k) + 1) * cycle_num
        return part1 + part2 + part3

class Enumbertransformationbootcamp(Basebootcamp):
    def __init__(self, max_b=10**18, max_a_diff=10**18, **kwargs):
        super().__init__(**kwargs)
        self.max_b = max_b
        self.max_a_diff = max_a_diff
    
    def case_generator(self):
        k = random.randint(2, 15)
        # 确保生成的b不超过1e18且>=1
        max_b_valid = min(self.max_b, 10**18)
        b = random.randint(1, max_b_valid)
        # 确保a不超过1e18
        max_valid_diff = min(self.max_a_diff, 10**18 - b)
        a_diff = random.randint(0, max_valid_diff)
        a = b + a_diff
        return {'a': a, 'b': b, 'k': k}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        a = question_case['a']
        b = question_case['b']
        k = question_case['k']
        return f"""你需要将正整数{a}转换为{b}。每次操作可选：
1. 减1，耗时1秒。
2. 选择x（2≤x≤{k}），将a替换为a - (a mod x)，耗时1秒。
计算所需最少秒数，并将答案放在[answer]标签内。例如：[answer]5[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity.get('a', 0)
        b = identity.get('b', 0)
        k = identity.get('k', 0)
        # 严格验证输入参数合法性
        if not (1 <= b <= a <= 10**18) or not (2 <= k <=15):
            return False
        try:
            correct = solve(a, b, k)
            return solution == correct
        except:
            return False
