"""# 

### 谜题描述
While creating high loaded systems one should pay a special attention to caching. This problem will be about one of the most popular caching algorithms called LRU (Least Recently Used).

Suppose the cache may store no more than k objects. At the beginning of the workflow the cache is empty. When some object is queried we check if it is present in the cache and move it here if it's not. If there are more than k objects in the cache after this, the least recently used one should be removed. In other words, we remove the object that has the smallest time of the last query.

Consider there are n videos being stored on the server, all of the same size. Cache can store no more than k videos and caching algorithm described above is applied. We know that any time a user enters the server he pick the video i with probability pi. The choice of the video is independent to any events before.

The goal of this problem is to count for each of the videos the probability it will be present in the cache after 10100 queries.

Input

The first line of the input contains two integers n and k (1 ≤ k ≤ n ≤ 20) — the number of videos and the size of the cache respectively. Next line contains n real numbers pi (0 ≤ pi ≤ 1), each of them is given with no more than two digits after decimal point.

It's guaranteed that the sum of all pi is equal to 1.

Output

Print n real numbers, the i-th of them should be equal to the probability that the i-th video will be present in the cache after 10100 queries. You answer will be considered correct if its absolute or relative error does not exceed 10 - 6. 

Namely: let's assume that your answer is a, and the answer of the jury is b. The checker program will consider your answer correct, if <image>.

Examples

Input

3 1
0.3 0.2 0.5


Output

0.3 0.2 0.5 

Input

2 1
0.0 1.0


Output

0.0 1.0 

Input

3 2
0.3 0.2 0.5


Output

0.675 0.4857142857142857 0.8392857142857143 

Input

3 3
0.2 0.3 0.5


Output

1.0 1.0 1.0 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <typename T>
void read(T &x);
template <typename T>
void write(T x);
template <typename T>
void writesp(T x);
template <typename T>
void writeln(T x);
const int N = 1ull << 21, M = 22;
double f[N];
int n, s, k;
double P[M], sum[N], p[M];
inline int pop_count(int x) {
  int res = 0;
  while (x) {
    if (x & 1) ++res;
    x >>= 1;
  }
  return res;
}
double res[M];
int cnt, pos[M];
int main() {
  read(n);
  read(k);
  for (register int i = 1; i <= n; i++) scanf(\"%lf\", &P[i]);
  for (register int i = 1; i <= n; i++) {
    if (P[i] > .0) p[++cnt] = P[i], pos[cnt] = i;
  }
  (k = min(k, cnt));
  s = (1ll << cnt);
  for (register int i = 1; i < s; i++)
    for (register int j = 0; j < cnt; j++)
      if (i >> j & 1) sum[i] += p[j + 1];
  f[0] = 1.0;
  for (register int i = 1; i < s; i++) {
    for (register int j = 0; j < cnt; j++) {
      if ((i >> j & 1))
        f[i] += f[i ^ (1 << j)] * p[j + 1] / (1.0 - sum[i ^ (1 << j)]);
    }
  }
  for (register int i = 1; i < s; i++) {
    if (pop_count(i) == k) {
      for (register int j = 0; j < cnt; j++) {
        if (i >> j & 1) res[pos[j + 1]] += f[i];
      }
    }
  }
  for (register int i = 1; i <= n; i++) printf(\"%.6f \", res[i]);
  putchar('\n');
}
template <typename T>
void read(T &x) {
  x = 0;
  int t = 1;
  char wn = getchar();
  while (wn < '0' || wn > '9') {
    if (wn == '-') t = -1;
    wn = getchar();
  }
  while (wn >= '0' && wn <= '9') {
    x = x * 10 + wn - '0';
    wn = getchar();
  }
  x *= t;
}
template <typename T>
void write(T x) {
  if (x < 0) {
    putchar('-');
    x = -x;
  }
  if (x <= 9) {
    putchar(x + '0');
    return;
  }
  write(x / 10);
  putchar(x % 10 + '0');
}
template <typename T>
void writesp(T x) {
  write(x);
  putchar(' ');
}
template <typename T>
void writeln(T x) {
  write(x);
  putchar('\n');
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_expected_probabilities(n, k, p_list):
    non_zero = [(idx, p) for idx, p in enumerate(p_list) if p > 1e-9]
    cnt = len(non_zero)
    real_k = min(k, cnt)
    
    if real_k >= cnt:
        expected = [0.0] * n
        for idx, p in non_zero:
            expected[idx] = 1.0
        return expected
    
    s = 1 << cnt
    sum_state = [0.0] * s
    for state in range(s):
        total = 0.0
        for j in range(cnt):
            if state & (1 << j):
                total += non_zero[j][1]
        sum_state[state] = total
    
    f = [0.0] * s
    f[0] = 1.0
    
    for state in range(1, s):
        for j in range(cnt):
            if state & (1 << j):
                prev = state ^ (1 << j)
                denominator = 1.0 - sum_state[prev]
                if denominator < 1e-9:
                    continue
                f[state] += f[prev] * non_zero[j][1] / denominator
    
    expected = [0.0] * n
    for state in range(s):
        pc = bin(state).count('1')
        if pc == real_k:
            for j in range(cnt):
                if state & (1 << j):
                    idx = non_zero[j][0]
                    expected[idx] += f[state]
    
    return expected

class Clrubootcamp(Basebootcamp):
    def __init__(self, max_n=20, **kwargs):
        super().__init__(**kwargs)
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        k = random.randint(1, n)
        
        # 生成和为100的整数分割（确保两位小数）
        total = 100
        parts = []
        for _ in range(n-1):
            part = random.randint(0, total)
            parts.append(part)
            total -= part
        parts.append(total)
        random.shuffle(parts)
        p = [x / 100.0 for x in parts]
        
        expected = compute_expected_probabilities(n, k, p)
        return {
            'n': n,
            'k': k,
            'p': p,
            'expected': expected
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        p = question_case['p']
        p_str = ' '.join(f"{pi:.2f}" for pi in p)
        prompt = (
            "You are tasked with calculating the probability of each video being present in an LRU (Least Recently Used) cache after 10^100 queries. The cache can hold up to k videos. Each query independently selects a video according to the given probabilities. Determine the steady-state probability for each video.\n\n"
            "Input format:\n"
            "- The first line contains two integers n and k (1 ≤ k ≤ n ≤ 20).\n"
            "- The second line contains n real numbers p1 p2 ... pn (sums to 1, each with up to two decimals).\n\n"
            "Output format:\n"
            "- Space-separated real numbers, each up to 15 decimal places, ensuring absolute/relative error ≤ 1e-6.\n\n"
            "Example Input:\n"
            "3 2\n"
            "0.30 0.20 0.50\n\n"
            "Example Output:\n"
            "0.675 0.485714285714286 0.839285714285714\n\n"
            "Your task:\n"
            f"{n} {k}\n"
            f"{p_str}\n\n"
            "Put your final answer within [answer] and [/answer], e.g., [answer]0.1 0.2 0.7[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        return last_match
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = identity['expected']
            parts = solution.split()
            if len(parts) != len(expected):
                return False
            for s_part, e_val in zip(parts, expected):
                s_val = float(s_part)
                if not cls.is_close(s_val, e_val):
                    return False
            return True
        except:
            return False
    
    @staticmethod
    def is_close(a, b, rel_tol=1e-6, abs_tol=1e-6):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
