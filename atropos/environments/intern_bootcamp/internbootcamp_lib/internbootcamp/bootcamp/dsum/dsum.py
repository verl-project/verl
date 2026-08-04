"""# 

### 谜题描述
You are given n non-decreasing arrays of non-negative numbers. 

Vasya repeats the following operation k times: 

  * Selects a non-empty array. 
  * Puts the first element of the selected array in his pocket. 
  * Removes the first element from the selected array. 



Vasya wants to maximize the sum of the elements in his pocket.

Input

The first line contains two integers n and k (1 ≤ n, k ≤ 3 000): the number of arrays and operations.

Each of the next n lines contain an array. The first integer in each line is t_i (1 ≤ t_i ≤ 10^6): the size of the i-th array. The following t_i integers a_{i, j} (0 ≤ a_{i, 1} ≤ … ≤ a_{i, t_i} ≤ 10^8) are the elements of the i-th array.

It is guaranteed that k ≤ ∑_{i=1}^n t_i ≤ 10^6.

Output

Print one integer: the maximum possible sum of all elements in Vasya's pocket after k operations.

Example

Input


3 3
2 5 10
3 1 2 3
2 1 20


Output


26

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <class T>
inline void rd(T &x) {
  int fl = 1;
  x = 0;
  char c = getchar();
  for (; !isdigit(c); c = getchar())
    if (c == '-') fl = -fl;
  for (; isdigit(c); c = getchar()) x = (x << 3) + (x << 1) + c - '0';
  x *= fl;
}
template <class T>
inline void wr(T x) {
  if (x < 0) x = -x, putchar('-');
  if (x < 10) {
    putchar(x + '0');
    return;
  }
  int tmp[30] = {0}, cnt = 0;
  while (x) tmp[cnt++] = x % 10, x /= 10;
  for (register int i = cnt - 1; i >= 0; --i) putchar(tmp[i] + '0');
}
const int N = 3e3 + 5;
int n, k;
long long ans, t[N], s[N];
vector<long long> a[N], f;
inline void solve(int l, int r) {
  if (l == r) {
    long long sum = 0;
    for (register int i = 0; i < k && i < t[l]; ++i)
      sum += a[l][i], ans = max(ans, f[k - i - 1] + sum);
    return;
  }
  vector<long long> tmp = f;
  for (register int i = l; i <= ((l + r) >> 1); ++i)
    for (register int j = k; j >= t[i]; --j)
      f[j] = max(f[j - t[i]] + s[i], f[j]);
  solve(((l + r) >> 1) + 1, r);
  f = tmp;
  for (register int i = ((l + r) >> 1) + 1; i <= r; ++i)
    for (register int j = k; j >= t[i]; --j)
      f[j] = max(f[j - t[i]] + s[i], f[j]);
  solve(l, ((l + r) >> 1));
}
int main() {
  rd(n);
  rd(k);
  f.resize(k + 10);
  for (register int i = 1; i <= n; ++i) {
    rd(t[i]);
    long long x;
    for (register int j = 1; j <= t[i]; ++j)
      rd(x), a[i].push_back(x), s[i] += x;
  }
  solve(1, n);
  wr(ans);
  puts(\"\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def calculate_max_sum(n, k, arrays):
    prefix_sums = []
    for arr in arrays:
        s = [0]
        total = 0
        for num in arr:
            total += num
            s.append(total)
        prefix_sums.append(s)
    
    dp = [-float('inf')] * (k + 1)
    dp[0] = 0
    
    for prefix in prefix_sums:
        t_i = len(prefix) - 1
        new_dp = list(dp)
        for j in range(k, -1, -1):
            max_val = -float('inf')
            max_m = min(t_i, j)
            for m in range(0, max_m + 1):
                if dp[j - m] + prefix[m] > max_val:
                    max_val = dp[j - m] + prefix[m]
            if max_val > new_dp[j]:
                new_dp[j] = max_val
        dp = new_dp
    
    return dp[k] if dp[k] != -float('inf') else 0

class Dsumbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=50, t_i_min=1, t_i_max=100, k_max=300, element_step=100):
        self.n_min = n_min
        self.n_max = n_max
        self.t_i_min = t_i_min
        self.t_i_max = t_i_max
        self.k_max = k_max
        self.element_step = element_step
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        arrays = []
        total_elements = 0
        
        for _ in range(n):
            t_i = random.randint(self.t_i_min, self.t_i_max)
            arr = []
            current = 0
            # 增加零元素出现的概率
            if random.random() < 0.2:
                current = 0
            for _ in range(t_i):
                step = random.randint(0, self.element_step)
                current += step
                arr.append(current)
            arrays.append(arr)
            total_elements += t_i
        
        k = random.randint(1, min(total_elements, self.k_max))
        
        expected_sum = calculate_max_sum(n, k, arrays)
        return {
            "n": n,
            "k": k,
            "arrays": arrays,
            "expected_sum": expected_sum
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [f"{question_case['n']} {question_case['k']}"]
        for arr in question_case['arrays']:
            t_i = len(arr)
            elements = ' '.join(map(str, arr))
            input_lines.append(f"{t_i} {elements}")
        input_str = '\n'.join(input_lines)
        
        prompt = f"""You are given {question_case['n']} non-decreasing arrays of non-negative numbers. 

Vasya repeats the following operation {question_case['k']} times: 

- Select a non-empty array. 
- Take the first element of the selected array and add it to his pocket. 
- Remove the first element from the selected array. 

Vasya wants to maximize the sum of the elements in his pocket.

Input format:
The first line contains two integers n and k (1 ≤ n, k ≤ 3000).
The next n lines each describe an array. Each line starts with an integer t_i (1 ≤ t_i ≤ 1e6) followed by t_i non-decreasing non-negative integers (0 ≤ a_i1 ≤ ... ≤ a_it_i ≤ 1e8).

Output:
Print one integer: the maximum possible sum.

The specific input for your task is:
{input_str}

Please write your answer within [ANSWER] and [/ANSWER] tags. For example: [ANSWER]123[/ANSWER]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[ANSWER\]\s*(\d+)\s*\[/ANSWER\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_sum']
