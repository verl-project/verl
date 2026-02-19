"""# 

### 谜题描述
You are given an array a consisting of n non-negative integers. You have to choose a non-negative integer x and form a new array b of size n according to the following rule: for all i from 1 to n, b_i = a_i ⊕ x (⊕ denotes the operation [bitwise XOR](https://en.wikipedia.org/wiki/Bitwise_operation#XOR)).

An inversion in the b array is a pair of integers i and j such that 1 ≤ i < j ≤ n and b_i > b_j.

You should choose x in such a way that the number of inversions in b is minimized. If there are several options for x — output the smallest one.

Input

First line contains a single integer n (1 ≤ n ≤ 3 ⋅ 10^5) — the number of elements in a.

Second line contains n space-separated integers a_1, a_2, ..., a_n (0 ≤ a_i ≤ 10^9), where a_i is the i-th element of a.

Output

Output two integers: the minimum possible number of inversions in b, and the minimum possible value of x, which achieves those number of inversions.

Examples

Input


4
0 1 3 2


Output


1 0


Input


9
10 7 9 10 7 5 5 3 5


Output


4 14


Input


3
8 10 3


Output


0 8

Note

In the first sample it is optimal to leave the array as it is by choosing x = 0.

In the second sample the selection of x = 14 results in b: [4, 9, 7, 4, 9, 11, 11, 13, 11]. It has 4 inversions:

  * i = 2, j = 3; 
  * i = 2, j = 4; 
  * i = 3, j = 4; 
  * i = 8, j = 9. 



In the third sample the selection of x = 8 results in b: [0, 2, 11]. It has no inversions.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
void solve() {
  int n;
  cin >> n;
  vector<int> a(n);
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  vector<string> data(n);
  for (int i = 0; i <= 29; i++) {
    long long tes = pow(2, i);
    for (int j = 0; j < n; j++) {
      if ((tes & a[j]) == tes) {
        data[j].push_back('1');
      } else {
        data[j].push_back('0');
      }
    }
  }
  vector<vector<int>> run(1);
  for (int i = 0; i < n; i++) {
    run[0].push_back(i);
  }
  long long b = 0;
  int ans = 0;
  vector<vector<int>> split;
  for (int i = 29; i >= 0; i--) {
    long long sum1 = 0, sum0 = 0;
    for (vector<int> j : run) {
      vector<int> zero;
      vector<int> one;
      int cnt0 = 0, cnt1 = 0;
      for (int k : j) {
        if (data[k][i] == '1') {
          sum0 += cnt0;
          cnt1++;
          one.push_back(k);
        } else {
          sum1 += cnt1;
          cnt0++;
          zero.push_back(k);
        }
      }
      if (one.size() != 0) split.push_back(one);
      if (zero.size() != 0) split.push_back(zero);
    }
    if (sum1 > sum0) {
      ans += pow(2, i);
    }
    b += min(sum1, sum0);
    swap(split, run);
    split.clear();
  }
  cout << b << \" \" << ans;
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  solve();
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random

def compute_min_inversion_xor(a):
    n = len(a)
    data = []
    for num in a:
        bits = []
        for i in range(30):
            if num & (1 << i):
                bits.append('1')
            else:
                bits.append('0')
        data.append(bits[::-1])  # Reverse to store MSB first
    
    run = [list(range(n))]
    ans = 0
    b_val = 0
    
    for i in range(30):  # Now using correct bit order
        sum0 = 0
        sum1 = 0
        split = []
        for group in run:
            zero = []
            one = []
            cnt0 = 0
            cnt1 = 0
            for k in group:
                if data[k][i] == '1':
                    sum0 += cnt0
                    cnt1 += 1
                    one.append(k)
                else:
                    sum1 += cnt1
                    cnt0 += 1
                    zero.append(k)
            if zero:
                split.append(zero)
            if one:
                split.append(one)
        if sum1 > sum0:
            ans += (1 << (29 - i))  # Adjust for bit significance
            b_val += sum0
        else:
            b_val += sum1
        run = split
    
    return b_val, ans

class Exorinversebootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10, max_value=10**9):
        self.n_min = n_min
        self.n_max = n_max
        self.max_value = max_value
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        a = [random.randint(0, self.max_value) for _ in range(n)]
        correct_inversions, correct_x = compute_min_inversion_xor(a)
        return {
            'n': n,
            'a': a,
            'correct_inversions': correct_inversions,
            'correct_x': correct_x
        }
    
    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        n = question_case['n']
        prompt = f"""You are given an array of {n} non-negative integers. Your task is to choose a non-negative integer x such that the array b, formed by XORing each element of the array with x, has the minimum number of inversions. An inversion is a pair of indices i < j where b[i] > b[j]. If multiple x values yield the same minimal number of inversions, choose the smallest x.

Input:
The first line contains an integer n ({n} in this case).
The second line contains {n} space-separated integers: {', '.join(map(str, a))}.

Your goal is to determine the minimal number of inversions and the corresponding smallest x.

Output two integers: the minimal number of inversions and the smallest x.

Please provide your answer within [answer] and [/answer]. For example:
[answer]42 7[/answer]

Your answer should be two integers separated by a space, enclosed within the tags."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        parts = last_match.split()
        if len(parts) != 2:
            return None
        try:
            inv = int(parts[0])
            x = int(parts[1])
            return (inv, x)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_inv = identity['correct_inversions']
        correct_x = identity['correct_x']
        return solution == (correct_inv, correct_x)
