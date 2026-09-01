"""# 

### 谜题描述
Little Petya loves training spiders. Petya has a board n × m in size. Each cell of the board initially has a spider sitting on it. After one second Petya chooses a certain action for each spider, and all of them humbly perform its commands. There are 5 possible commands: to stay idle or to move from current cell to some of the four side-neighboring cells (that is, one command for each of the four possible directions). Petya gives the commands so that no spider leaves the field. It is allowed for spiders to pass through each other when they crawl towards each other in opposite directions. All spiders crawl simultaneously and several spiders may end up in one cell. Petya wants to know the maximum possible number of spider-free cells after one second.

Input

The first line contains two space-separated integers n and m (1 ≤ n, m ≤ 40, n·m ≤ 40) — the board sizes.

Output

In the first line print the maximum number of cells without spiders.

Examples

Input

1 1


Output

0


Input

2 3


Output

4

Note

In the first sample the only possible answer is:

s

In the second sample one of the possible solutions is: 
    
    
      
    rdl  
    rul  
    

s denotes command \"stay idle\", l, r, d, u denote commands \"crawl left\", \"crawl right\", \"crawl down\", \"crawl up\", correspondingly.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n, m;
int dp[45][64][64];
int ans;
bool get_bit(int a, int n) { return (a >> n) & 1; }
int set_bit(int a, int n) { return a | (1 << n); }
int reset_bit(int a, int n) { return a & ~(1 << n); }
int main() {
  cin >> n >> m;
  if (m > n) swap(n, m);
  for (int i = 0; i <= n; i++) {
    for (int j = 0; j < (1 << m); j++) {
      for (int k = 0; k < (1 << m); k++) {
        dp[i][j][k] = -1000;
      }
    }
  }
  dp[0][0][(1 << m) - 1] = 0;
  for (int i = 1; i <= n; i++) {
    for (int j = 0; j < (1 << m); j++) {
      for (int k = 0; k < (1 << m); k++) {
        for (int l = 0; l < (1 << m); l++) {
          int s = j | l;
          int cnt = 0;
          int ns = (1 << m) - 1;
          int nr = (1 << m) - 1 - s;
          for (int i = 0; i < m; i++) {
            if (!get_bit(s, i)) {
              cnt++;
            }
          }
          if (m == 1) {
            if (get_bit(s, 0)) {
              ns = reset_bit(ns, 0);
            }
          } else {
            if (get_bit(s, 0)) {
              ns = reset_bit(ns, 0);
              ns = reset_bit(ns, 1);
            }
            if (get_bit(s, m - 1)) {
              ns = reset_bit(ns, m - 1);
              ns = reset_bit(ns, m - 2);
            }
            for (int i = 1; i < m - 1; i++) {
              if (get_bit(s, i)) {
                ns = reset_bit(ns, i - 1);
                ns = reset_bit(ns, i);
                ns = reset_bit(ns, i + 1);
              }
            }
          }
          dp[i][ns & k][nr] = max(dp[i][ns & k][nr], dp[i - 1][j][k] + cnt);
        }
      }
    }
  }
  for (int i = 0; i < (1 << m); i++) ans = max(ans, dp[n][0][i]);
  cout << ans;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def get_bit(a, n):
    return (a >> n) & 1

def reset_bit(a, n):
    return a & ~(1 << n)

def calculate_max_empty(n, m):
    # Ensure n is the larger dimension for optimization
    if m > n:
        n, m = m, n
    if m == 0:
        return 0  # Should not happen for valid input
    max_size = 1 << m
    dp = [[[-1000] * max_size for _ in range(max_size)] for __ in range(n + 1)]
    initial_mask = (1 << m) - 1
    dp[0][0][initial_mask] = 0
    
    for i in range(1, n + 1):
        for prev_row in range(max_size):
            for prev_mask in range(max_size):
                if dp[i-1][prev_row][prev_mask] == -1000:
                    continue
                for current_row in range(max_size):
                    # Calculate spiders present in current configuration
                    combined = prev_row | current_row
                    cnt = sum(1 for bit in range(m) if not get_bit(combined, bit))
                    
                    # Calculate new_mask based on spider movements
                    new_mask = initial_mask
                    for bit in range(m):
                        if get_bit(combined, bit):
                            if m == 1:
                                new_mask = reset_bit(new_mask, 0)
                            else:
                                for offset in (-1, 0, 1):
                                    pos = bit + offset
                                    if 0 <= pos < m:
                                        new_mask = reset_bit(new_mask, pos)
                    
                    next_mask = new_mask & prev_mask
                    dp[i][next_mask][current_row] = max(
                        dp[i][next_mask][current_row], 
                        dp[i-1][prev_row][prev_mask] + cnt
                    )
    
    # Find maximum value in final state
    return max(dp[n][0][state] for state in range(max_size))

class Epetyaandspidersbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.params = params
        self.params.setdefault('n', None)
        self.params.setdefault('m', None)
    
    def case_generator(self):
        # Generate valid (n, m) pairs where 1 ≤ n,m ≤40 and n*m ≤40
        n = self.params['n']
        m = self.params['m']
        
        if n is None or m is None:
            possible = []
            for a in range(1, 41):
                for b in range(1, 41):
                    if a * b <= 40:
                        possible.append((a, b))
            if possible:
                n, m = random.choice(possible)
            else:
                n, m = 1, 1
        
        answer = calculate_max_empty(n, m)
        return {'n': n, 'm': m, 'answer': answer}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        return f"""You are a programming competition expert. Solve the following problem:

Petya has a board of size {n} rows × {m} columns. Each cell starts with a spider. All spiders simultaneously move to adjacent cells or stay in place for 1 second. Find the maximum number of empty cells possible after exactly one second.

Movement rules:
1. Epetyaandspiderss cannot move outside the board
2. Multiple spiders can occupy the same cell
3. Valid moves: stay (s), left (l), right (r), up (u), down (d)

Examples:
Input: 1 1 → Output: 0
Input: 2 3 → Output: 4

Your task: Given a {n}x{m} grid, compute the maximum possible empty cells after one second.

Format your answer as [answer]NUMBER[/answer]. Example: [answer]4[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except (ValueError, IndexError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
