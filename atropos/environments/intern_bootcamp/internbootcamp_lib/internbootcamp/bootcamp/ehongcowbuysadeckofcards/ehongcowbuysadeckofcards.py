"""# 

### 谜题描述
One day, Hongcow goes to the store and sees a brand new deck of n special cards. Each individual card is either red or blue. He decides he wants to buy them immediately. To do this, he needs to play a game with the owner of the store.

This game takes some number of turns to complete. On a turn, Hongcow may do one of two things: 

  * Collect tokens. Hongcow collects 1 red token and 1 blue token by choosing this option (thus, 2 tokens in total per one operation). 
  * Buy a card. Hongcow chooses some card and spends tokens to purchase it as specified below. 



The i-th card requires ri red resources and bi blue resources. Suppose Hongcow currently has A red cards and B blue cards. Then, the i-th card will require Hongcow to spend max(ri - A, 0) red tokens, and max(bi - B, 0) blue tokens. Note, only tokens disappear, but the cards stay with Hongcow forever. Each card can be bought only once.

Given a description of the cards and their costs determine the minimum number of turns Hongcow needs to purchase all cards.

Input

The first line of input will contain a single integer n (1 ≤ n ≤ 16).

The next n lines of input will contain three tokens ci, ri and bi. ci will be 'R' or 'B', denoting the color of the card as red or blue. ri will be an integer denoting the amount of red resources required to obtain the card, and bi will be an integer denoting the amount of blue resources required to obtain the card (0 ≤ ri, bi ≤ 107).

Output

Output a single integer, denoting the minimum number of turns needed to acquire all the cards.

Examples

Input

3
R 0 1
B 1 0
R 1 1


Output

4


Input

3
R 3 0
R 2 0
R 1 0


Output

6

Note

For the first sample, Hongcow's four moves are as follows: 

  1. Collect tokens 
  2. Buy card 1
  3. Buy card 2
  4. Buy card 3

Note, at the fourth step, Hongcow is able to buy card 3 because Hongcow already has one red and one blue card, so we don't need to collect tokens.

For the second sample, one optimal strategy is as follows: 

  1. Collect tokens 
  2. Collect tokens 
  3. Buy card 2
  4. Collect tokens 
  5. Buy card 3
  6. Buy card 1

At the fifth step, even though Hongcow has a red token, Hongcow doesn't actually need to spend it, since Hongcow has a red card already.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long MOD = 1e9 + 7;
const int INF = 2e9;
const long long INF64 = 3e18;
const double EPS = 1e-9;
const double PI = acos(-1);
const long long MD = 1551513443;
const long long T = 25923;
const int N = 100001;
const int M = 16;
const bool DEBUG = 1;
int n, r[M], b[M], dp[1 << M][M * M + 1][2];
pair<int, int> z[1 << M];
char color[M];
int main() {
  cin >> n;
  int s0 = 0, s1 = 0;
  for (int i = 0; i < int(n); i++) {
    cin >> color[i] >> r[i] >> b[i];
    s0 += max(0, r[i] - n);
    r[i] = min(r[i], n);
    s1 += max(0, b[i] - n);
    b[i] = min(b[i], n);
  }
  for (int mask = 0; mask < int((1 << n)); mask++)
    for (int i = 0; i < int(n); i++)
      if (mask & (1 << i)) {
        if (color[i] == 'R')
          z[mask].first++;
        else
          z[mask].second++;
      }
  for (int i = 0; i < int((1 << n)); i++)
    for (int j = 0; j < int(n * n + 1); j++)
      for (int t = 0; t < int(2); t++) dp[i][j][t] = INF;
  dp[0][0][0] = dp[0][0][1] = 0;
  for (int mask = 0; mask < int((1 << n)); mask++)
    for (int u = 0; u < int(n * n + 1); u++)
      for (int t = 0; t < int(2); t++)
        if (dp[mask][u][t] != INF) {
          for (int i = 0; i < int(n); i++)
            if (!(mask & (1 << i))) {
              int new_mask = mask + (1 << i), xx = 0, yy = 0;
              if (t == 0)
                xx = u;
              else
                yy = u;
              int first = max(0, r[i] - z[mask].first),
                  second = max(0, b[i] - z[mask].second);
              xx -= first;
              yy -= second;
              int q = -min(xx, yy);
              xx += q;
              yy += q;
              if (yy == 0)
                dp[new_mask][xx][0] =
                    min(dp[new_mask][xx][0], dp[mask][u][t] + q);
              else
                dp[new_mask][yy][1] =
                    min(dp[new_mask][yy][1], dp[mask][u][t] + q);
            }
        }
  int ans = INF, mask = (1 << n) - 1;
  for (int u = 0; u < int(n * n + 1); u++)
    for (int t = 0; t < int(2); t++)
      if (dp[mask][u][t] != INF) {
        int xx = 0, yy = 0;
        if (t == 0)
          xx = u;
        else
          yy = u;
        ans = min(ans, max(max(0, s0 - xx), max(0, s1 - yy)) + dp[mask][u][t]);
      }
  cout << ans + n << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_min_turns(n, cards):
    s0 = 0
    s1 = 0
    color = []
    r = []
    b = []
    for ci, ri, bi in cards:
        color.append(ci)
        new_ri = min(ri, n)
        s0 += max(ri - new_ri, 0)
        r.append(new_ri)
        new_bi = min(bi, n)
        s1 += max(bi - new_bi, 0)
        b.append(new_bi)
    
    max_mask = 1 << n
    z = [(0, 0)] * max_mask
    for mask in range(max_mask):
        red = 0
        blue = 0
        for i in range(n):
            if mask & (1 << i):
                if color[i] == 'R':
                    red += 1
                else:
                    blue += 1
        z[mask] = (red, blue)
    
    INF = float('inf')
    max_u = n * n
    dp = [[[INF] * 2 for _ in range(max_u + 1)] for __ in range(max_mask)]
    dp[0][0][0] = 0
    dp[0][0][1] = 0
    
    for mask in range(max_mask):
        for u in range(max_u + 1):
            for t in [0, 1]:
                if dp[mask][u][t] == INF:
                    continue
                for i in range(n):
                    if not (mask & (1 << i)):
                        current_red, current_blue = z[mask]
                        needed_red = max(r[i] - current_red, 0)
                        needed_blue = max(b[i] - current_blue, 0)
                        
                        if t == 0:
                            xx = u - needed_red
                            yy = -needed_blue
                        else:
                            xx = -needed_red
                            yy = u - needed_blue
                        
                        # 正确的q计算
                        q = -min(xx, yy)
                        xx += q
                        yy += q
                        
                        # 确保状态合法
                        if xx < 0 or yy < 0:
                            continue
                        
                        next_t = 0 if yy == 0 else 1
                        next_u = xx if next_t == 0 else yy
                        if next_u > max_u:
                            continue
                        
                        new_mask = mask | (1 << i)
                        if dp[new_mask][next_u][next_t] > dp[mask][u][t] + q:
                            dp[new_mask][next_u][next_t] = dp[mask][u][t] + q
    
    ans = INF
    full_mask = (1 << n) - 1
    for u in range(max_u + 1):
        for t in [0, 1]:
            if dp[full_mask][u][t] != INF:
                if t == 0:
                    xx = u
                    yy = 0
                else:
                    xx = 0
                    yy = u
                total_red = max(s0 - xx, 0)
                total_blue = max(s1 - yy, 0)
                total_add = max(total_red, total_blue)
                total_steps = dp[full_mask][u][t] + total_add
                if total_steps < ans:
                    ans = total_steps
    return ans + n if ans != INF else 0

class Ehongcowbuysadeckofcardsbootcamp(Basebootcamp):
    def __init__(self, max_n=5, r_max=10, b_max=10):
        self.max_n = max_n
        self.r_max = r_max
        self.b_max = b_max
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        cards = []
        for _ in range(n):
            color = random.choice(['R', 'B'])
            r = random.randint(0, self.r_max)
            b = random.randint(0, self.b_max)
            cards.append((color, r, b))
        answer = compute_min_turns(n, cards)
        return {
            'n': n,
            'cards': cards,
            'answer': answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        cards_desc = '\n'.join([' '.join(map(str, card)) for card in question_case['cards']])
        return f'''Hongcow needs to buy all cards from the store. Each card is Red (R) or Blue (B) and requires tokens:
Input:
{question_case['n']}
{cards_desc}

Rules:
1. Each turn: collect tokens (gain 1R+1B) or buy a card (spend required tokens)
2. To buy card i:
   - R tokens needed: max(r_i - owned_R_cards, 0)
   - B tokens needed: max(b_i - owned_B_cards, 0)
3. Cards permanently reduce future requirements

Output the minimal turns. Put your answer within [answer]...[/answer]. Example: [answer]6[/answer]'''

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answer')
