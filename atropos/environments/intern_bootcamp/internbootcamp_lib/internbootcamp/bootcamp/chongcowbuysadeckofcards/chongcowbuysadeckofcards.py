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
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0), cout.tie(0);
  cout << fixed << setprecision(15);
  int N;
  cin >> N;
  vector<bool> color(N);
  vector<int> r(N), b(N);
  for (int i = 0; i < N; i++) {
    string t;
    cin >> t >> r[i] >> b[i];
    color[i] = (t == \"B\");
  }
  int MSAVE = 0;
  for (int i = 0; i < N; i++) MSAVE += i;
  vector<vector<int> > dp(1 << N, vector<int>(MSAVE + 1, -1));
  dp[0][0] = 0;
  for (int m = 0; m < (1 << N); m++) {
    int red = 0, blue = 0;
    for (int i = 0; i < N; i++) {
      if ((m >> i) & 1) {
        if (color[i])
          blue++;
        else
          red++;
      }
    }
    for (int rs = 0; rs <= MSAVE; rs++) {
      if (dp[m][rs] == -1) continue;
      for (int t = 0; t < N; t++) {
        if ((m >> t) & 1) continue;
        int nrs = rs, nbs = dp[m][rs];
        nbs += min(blue, b[t]);
        nrs += min(red, r[t]);
        int &upd = dp[m ^ (1 << t)][nrs];
        upd = max(upd, nbs);
      }
    }
  }
  int blue_need = 0, red_need = 0;
  for (int c : r) red_need += c;
  for (int c : b) blue_need += c;
  int ans = max(blue_need, red_need);
  int fm = (1 << N) - 1;
  for (int rs = 0; rs <= MSAVE; rs++)
    if (dp[fm][rs] != -1) {
      ans = min(ans, max(blue_need - dp[fm][rs], red_need - rs));
    }
  cout << ans + N << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Chongcowbuysadeckofcardsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=8, max_r=5, max_b=5, **kwargs):
        super().__init__(**kwargs)
        # 参数校验确保合理范围
        self.min_n = max(1, min(min_n, 10))
        self.max_n = max(self.min_n, min(max_n, 10))
        self.max_r = min(max(1, max_r), 20)
        self.max_b = min(max(1, max_b), 20)

    def case_generator(self):
        for _ in range(10):  # 最多尝试10次生成有效案例
            n = random.randint(self.min_n, self.max_n)
            cards = [{
                'color': random.choice(['R', 'B']),
                'r': random.randint(0, self.max_r),
                'b': random.randint(0, self.max_b)
            } for _ in range(n)]
            
            # 确保至少一个卡片有需求
            if all(c['r'] == 0 and c['b'] == 0 for c in cards):
                continue
            
            try:
                answer = self.calculate_min_turns(n, cards)
                return {
                    'n': n,
                    'cards': cards,
                    'correct_answer': answer
                }
            except Exception as e:
                continue
        
        # 保底生成简单案例
        return {
            'n': 2,
            'cards': [
                {'color': 'R', 'r': 1, 'b': 0},
                {'color': 'B', 'r': 0, 'b': 1}
            ],
            'correct_answer': 2
        }

    @staticmethod
    def prompt_func(question_case):
        case_str = "\n".join(
            f"{card['color']} {card['r']} {card['b']}" 
            for card in question_case['cards']
        )
        return f"""Hongcow需要购买所有卡片。每个操作可以：
1. 收集红蓝令牌各1个
2. 购买卡片（需消耗max(需求-已有对应颜色卡数,0)的对应令牌）

输入：
{question_case['n']}
{case_str}

请计算最少操作次数，并将答案放在[answer]标签内。示例：[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        try:
            return int(matches[-1].strip()) if matches else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']

    @staticmethod
    def calculate_min_turns(n, cards):
        # 预处理卡片数据
        color = [1 if c['color'] == 'B' else 0 for c in cards]
        r = [c['r'] for c in cards]
        b = [c['b'] for c in cards]
        
        total_r = sum(r)
        total_b = sum(b)
        max_rsave = total_r  # 红令牌最多能节省的总量
        
        # DP状态定义：dp[mask][rsave] = 最大bsave
        dp = [[-1]*(max_rsave+1) for _ in range(1<<n)]
        dp[0][0] = 0  # 初始状态
        
        for mask in range(1<<n):
            # 计算当前拥有的红蓝卡数量
            current_r = sum(0 if color[i] else 1 
                          for i in range(n) if (mask >> i) & 1)
            current_b = sum(1 if color[i] else 0 
                          for i in range(n) if (mask >> i) & 1)
            
            for rsave in range(max_rsave+1):
                if dp[mask][rsave] == -1:
                    continue
                
                # 尝试购买下一张卡片
                for next_card in range(n):
                    if (mask & (1 << next_card)) == 0:
                        # 计算实际需要支付的令牌
                        needed_r = max(r[next_card] - current_r, 0)
                        needed_b = max(b[next_card] - current_b, 0)
                        
                        # 累计节省的令牌
                        new_rsave = rsave + (r[next_card] - needed_r)
                        new_bsave = dp[mask][rsave] + (b[next_card] - needed_b)
                        new_mask = mask | (1 << next_card)
                        
                        # 更新状态
                        if new_rsave <= max_rsave and new_bsave > dp[new_mask][new_rsave]:
                            dp[new_mask][new_rsave] = new_bsave
        
        # 计算最终结果
        min_ops = max(total_r, total_b)  # 初始值
        full_mask = (1 << n) - 1
        
        for rsave in range(max_rsave+1):
            if dp[full_mask][rsave] != -1:
                required_r = max(total_r - rsave, 0)
                required_b = max(total_b - dp[full_mask][rsave], 0)
                min_ops = min(min_ops, max(required_r, required_b))
        
        return min_ops + n  # 加上购买卡片的n次操作
