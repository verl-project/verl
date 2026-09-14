"""# 

### 谜题描述
The Little Elephant has two permutations a and b of length n, consisting of numbers from 1 to n, inclusive. Let's denote the i-th (1 ≤ i ≤ n) element of the permutation a as ai, the j-th (1 ≤ j ≤ n) element of the permutation b — as bj.

The distance between permutations a and b is the minimum absolute value of the difference between the positions of the occurrences of some number in a and in b. More formally, it's such minimum |i - j|, that ai = bj.

A cyclic shift number i (1 ≤ i ≤ n) of permutation b consisting from n elements is a permutation bibi + 1... bnb1b2... bi - 1. Overall a permutation has n cyclic shifts.

The Little Elephant wonders, for all cyclic shifts of permutation b, what is the distance between the cyclic shift and permutation a?

Input

The first line contains a single integer n (1 ≤ n ≤ 105) — the size of the permutations. The second line contains permutation a as n distinct numbers from 1 to n, inclusive. The numbers are separated with single spaces. The third line contains permutation b in the same format.

Output

In n lines print n integers — the answers for cyclic shifts. Print the answers to the shifts in the order of the shifts' numeration in permutation b, that is, first for the 1-st cyclic shift, then for the 2-nd, and so on.

Examples

Input

2
1 2
2 1


Output

1
0


Input

4
2 1 3 4
3 4 2 1


Output

2
1
0
1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 100010;
int n, a[MAXN], b[MAXN], ia[MAXN], ans[MAXN];
priority_queue<pair<int, int> > pq1, pq2;
int main() {
  if (fopen(\"input.txt\", \"r\")) {
    freopen(\"input.txt\", \"r\", stdin);
  }
  scanf(\"%d\", &n);
  for (int i = 0; i < n; i++) {
    scanf(\"%d\", &a[i]);
    a[i]--;
    ia[a[i]] = i;
  }
  for (int i = 0; i < n; i++) {
    scanf(\"%d\", &b[i]);
    b[i]--;
  }
  for (int i = 0; i < n; i++) ans[i] = 1e9;
  for (int i = 0; i < n; i++) {
    if (i <= ia[b[i]]) {
      pq1.push(pair<int, int>(-(ia[b[i]] - i), i));
    } else {
      pq2.push(pair<int, int>(-(i - ia[b[i]]), i));
    }
    ans[0] = min(ans[0], abs(i - ia[b[i]]));
  }
  for (int i = 1; i < n; i++) {
    pq2.push(pair<int, int>(-(n - ia[b[i - 1]] - 1 + i), n + i));
    while (!pq2.empty() && -pq2.top().first - i < 0) {
      pq1.push(pair<int, int>(-(1 - i), pq2.top().second));
      pq2.pop();
    }
    while (!pq1.empty() && pq1.top().second - i < 0) pq1.pop();
    if (!pq1.empty()) ans[i] = min(ans[i], -pq1.top().first + i);
    if (!pq2.empty()) ans[i] = min(ans[i], -pq2.top().first - i);
  }
  for (int i = 0; i < n; i++) printf(\"%d\n\", ans[i]);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from heapq import heappop, heappush
from bootcamp import Basebootcamp

class Clittleelephantandshiftsbootcamp(Basebootcamp):
    def __init__(self, n=5):
        self.n = n
    
    def case_generator(self):
        n = self.n
        a = list(range(1, n+1))
        random.shuffle(a)
        b = list(range(1, n+1))
        random.shuffle(b)
        expected_output = self.compute_expected(n, a, b)
        return {
            'n': n,
            'a': a,
            'b': b,
            'expected_output': expected_output
        }
    
    @staticmethod
    def compute_expected(n, a, b):
        # Convert to 0-based and precompute positions in a
        a_pos = {num: idx for idx, num in enumerate(a)}
        ia = [0] * n
        for idx, num in enumerate(a):
            ia[num-1] = idx  # since a contains 1-based numbers
        
        # Convert b to 0-based indices in b list
        b_zero = [num-1 for num in b]  # to 0-based internally
        
        ans = [float('inf')] * n
        # Priority queues store (-distance, original index)
        pq_left = []  # elements where i <= ia[b[i]]
        pq_right = []  # elements where i > ia[b[i]]
        
        for i in range(n):
            current_b = b_zero[i]
            pos_in_a = ia[current_b]
            diff = i - pos_in_a
            if i <= pos_in_a:
                heappush(pq_left, (-(pos_in_a - i), i))
            else:
                heappush(pq_right, (-(i - pos_in_a), i))
            ans[0] = min(ans[0], abs(i - pos_in_a))
        
        for k in range(1, n):
            # Move elements from previous shift out of the window
            prev_idx = k - 1
            current_b_prev = b_zero[prev_idx]
            pos_in_a_prev = ia[current_b_prev]
            shifted_pos = (prev_idx - (k-1)) % n  # was considered for previous k-1 shifts
            
            new_diff_for_next = (n - pos_in_a_prev - 1) + k
            heappush(pq_right, (-new_diff_for_next, n + prev_idx))
            
            # Remove elements from pq_right that are now in pq_left due to shift
            while pq_right and -pq_right[0][0] - k < 0:
                dist, idx = heappop(pq_right)
                new_dist = - (-dist - k)
                heappush(pq_left, (-new_dist, idx))
            
            # Remove elements from pq_left that are out of the valid indices (>=k)
            while pq_left and pq_left[0][1] < k:
                heappop(pq_left)
            
            current_min = float('inf')
            if pq_left:
                current_min = min(current_min, -pq_left[0][0] + k)
            if pq_right:
                current_min = min(current_min, -pq_right[0][0] - k)
            
            ans[k] = current_min
        
        return ans
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        b = ' '.join(map(str, question_case['b']))
        prompt = f"""You are a programming competition contestant. Solve the following problem and present your answer in the required format.

Problem Statement:

The Little Elephant has two permutations a and b of length n, where n is a positive integer. The distance between two permutations is defined as the minimum absolute difference between the positions of a common element in a and in some cyclic shift of b. For each of the n cyclic shifts of permutation b, determine the distance to permutation a.

Input Format:

The input consists of:
- The first line contains an integer n, the size of the permutations.
- The second line contains permutation a as n distinct integers separated by spaces.
- The third line contains permutation b in the same format.

Output Format:

Output n lines, each containing one integer. The i-th line should correspond to the distance between permutation a and the i-th cyclic shift of permutation b.

Cyclic Shift Explanation:

A cyclic shift by i (1 ≤ i ≤ n) of permutation b is formed by taking the first i elements and moving them to the end. For example, if b is [3,4,2,1], the 1st cyclic shift is [3,4,2,1], the 2nd is [4,2,1,3], the 3rd is [2,1,3,4], and the 4th is [1,3,4,2].

Examples:

Sample Input 1:
2
1 2
2 1

Sample Output 1:
1
0

Sample Input 2:
4
2 1 3 4
3 4 2 1

Sample Output 2:
2
1
0
1

Your Task:

Compute the distance for each cyclic shift of b and output them in order. Ensure your answer has exactly n lines, each with a single integer. Place your final answer within [answer] tags, like this:

[answer]
<output line 1>
<output line 2>
...
[/answer]

Now, solve the following problem:

Input:
{n}
{a}
{b}

Your answer:
"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        try:
            solution = list(map(int, lines))
            return solution
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_output']
        if not isinstance(solution, list) or len(solution) != len(expected):
            return False
        return all(s == e for s, e in zip(solution, expected))
