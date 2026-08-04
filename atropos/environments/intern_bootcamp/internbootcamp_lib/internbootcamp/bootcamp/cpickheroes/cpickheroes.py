"""# 

### 谜题描述
Don't you tell me what you think that I can be

If you say that Arkady is a bit old-fashioned playing checkers, you won't be right. There is also a modern computer game Arkady and his friends are keen on. We won't discuss its rules, the only feature important to this problem is that each player has to pick a distinct hero in the beginning of the game.

There are 2 teams each having n players and 2n heroes to distribute between the teams. The teams take turns picking heroes: at first, the first team chooses a hero in its team, after that the second team chooses a hero and so on. Note that after a hero is chosen it becomes unavailable to both teams.

The friends estimate the power of the i-th of the heroes as p_i. Each team wants to maximize the total power of its heroes. However, there is one exception: there are m pairs of heroes that are especially strong against each other, so when any team chooses a hero from such a pair, the other team must choose the other one on its turn. Each hero is in at most one such pair.

This is an interactive problem. You are to write a program that will optimally choose the heroes for one team, while the jury's program will play for the other team. Note that the jury's program may behave inefficiently, in this case you have to take the opportunity and still maximize the total power of your team. Formally, if you ever have chance to reach the total power of q or greater regardless of jury's program choices, you must get q or greater to pass a test.

Input

The first line contains two integers n and m (1 ≤ n ≤ 10^3, 0 ≤ m ≤ n) — the number of players in one team and the number of special pairs of heroes.

The second line contains 2n integers p_1, p_2, …, p_{2n} (1 ≤ p_i ≤ 10^3) — the powers of the heroes.

Each of the next m lines contains two integer a and b (1 ≤ a, b ≤ 2n, a ≠ b) — a pair of heroes that are especially strong against each other. It is guaranteed that each hero appears at most once in this list.

The next line contains a single integer t (1 ≤ t ≤ 2) — the team you are to play for. If t = 1, the first turn is yours, otherwise you have the second turn.

Hacks

In order to hack, use the format described above with one additional line. In this line output 2n distinct integers from 1 to 2n — the priority order for the jury's team. The jury's team will on each turn select the first possible hero from this list. Here possible means that it is not yet taken and does not contradict the rules about special pair of heroes.

Interaction

When it is your turn, print a single integer x (1 ≤ x ≤ 2n) — the index of the hero chosen by you. Note that you can't choose a hero previously chosen by either you of the other player, and you must follow the rules about special pairs of heroes.

When it is the other team's turn, read a line containing a single integer x (1 ≤ x ≤ 2n) — the index of the hero chosen by the other team. It is guaranteed that this index is not chosen before and that the other team also follows the rules about special pairs of heroes.

After the last turn you should terminate without printing anything.

After printing your choice do not forget to output end of line and flush the output. Otherwise you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++;
  * System.out.flush() in Java;
  * flush(output) in Pascal;
  * stdout.flush() in Python;
  * see documentation for other languages.



Jury's answer -1 instead of a valid choice means that you made an invalid turn. Exit immediately after receiving -1 and you will see Wrong answer verdict. Otherwise you can get an arbitrary verdict because your solution will continue to read from a closed stream.

Examples

Input


3 1
1 2 3 4 5 6
2 6
1

2

4

1


Output






6

5

3


Input


3 1
1 2 3 4 5 6
1 5
2
6

1

3


Output







5

4

2

Note

In the first example the first turn is yours. In example, you choose 6, the other team is forced to reply with 2. You choose 5, the other team chooses 4. Finally, you choose 3 and the other team choose 1.

In the second example you have the second turn. The other team chooses 6, you choose 5, forcing the other team to choose 1. Now you choose 4, the other team chooses 3 and you choose 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, m;
  cin >> n >> m;
  set<pair<int, int>> ps;
  vector<int> po(2 * n), sc(2 * n);
  for (int i = 0; i < 2 * n; i++) {
    cin >> po[i];
    sc[i] = po[i];
  }
  map<int, int> opp;
  for (int i = 0; i < m; i++) {
    int a, b;
    cin >> a >> b;
    a--;
    b--;
    opp[a] = b;
    opp[b] = a;
    if (po[a] < po[b]) swap(a, b);
    sc[a] = 1e6 + (po[a] - po[b]);
  }
  for (int i = 0; i < 2 * n; i++) {
    ps.insert({-sc[i], i});
  }
  int t, la = -1;
  cin >> t;
  for (int i = 0; i < 2 * n; i++) {
    if (t == 1) {
      if (opp.find(la) != opp.end() and
          ps.find({-sc[opp[la]], opp[la]}) != ps.end())
        la = opp[la];
      else
        la = ps.begin()->second;
      cout << la + 1 << endl;
      cout.flush();
    } else {
      cin >> la;
      la--;
    }
    assert(ps.find({-sc[la], la}) != ps.end());
    ps.erase({-sc[la], la});
    t = 3 - t;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import heapq
import re
from bootcamp import Basebootcamp

class Cpickheroesbootcamp(Basebootcamp):
    def __init__(self, default_n=3, max_m=None, p_range=(1, 1000)):
        self.default_n = default_n
        self.max_m = default_n if max_m is None else max_m
        self.p_range = p_range
    
    def case_generator(self):
        n = self.default_n
        m = random.randint(0, self.max_m)
        p = [random.randint(*self.p_range) for _ in range(2 * n)]
        
        available = list(range(1, 2 * n + 1))
        pairs = []
        for _ in range(m):
            if len(available) < 2:
                break
            a = random.choice(available)
            available.remove(a)
            b = random.choice(available)
            available.remove(b)
            pairs.append([a, b])
        m = len(pairs)
        
        jury_priority = list(range(1, 2 * n + 1))
        random.shuffle(jury_priority)
        t = random.choice([1, 2])
        
        correct_total = self.simulate_case(n, m, p, pairs, t, jury_priority)
        
        return {
            'n': n,
            'm': m,
            'p': p,
            'pairs': pairs,
            't': t,
            'jury_priority': jury_priority,
            'correct_total': correct_total
        }
    
    def simulate_case(self, n, m, p, pairs, t, jury_priority):
        opp = {}
        for a, b in pairs:
            a_idx = a - 1
            b_idx = b - 1
            opp[a_idx] = b_idx
            opp[b_idx] = a_idx
        
        sc = p.copy()
        processed = set()
        for a, b in pairs:
            a_idx = a - 1
            b_idx = b - 1
            if a_idx in processed or b_idx in processed:
                continue
            a_val, b_val = p[a_idx], p[b_idx]
            if a_val < b_val:
                max_idx, min_idx = b_idx, a_idx
            else:
                max_idx, min_idx = a_idx, b_idx
            sc[max_idx] += 10**6 - p[max_idx]
            processed.update({a_idx, b_idx})
        
        available = set(range(2 * n))
        user_team = []
        jury_team = []
        turn = t
        la = -1
        
        jury_priority_idx = [x - 1 for x in jury_priority]
        
        for _ in range(2 * n):
            if turn == 1:
                if la != -1 and la in opp and opp[la] in available:
                    x = opp[la]
                else:
                    available_list = sorted(available, key=lambda i: (-sc[i], i))
                    x = available_list[0] if available_list else None
                user_team.append(x)
                available.remove(x)
                la = x
                turn = 2
            else:
                if la != -1 and la in opp and opp[la] in available:
                    x = opp[la]
                else:
                    x = next((h for h in jury_priority_idx if h in available), None)
                jury_team.append(x)
                available.remove(x)
                la = x
                turn = 1
        
        return sum(p[i] for i in user_team)
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        p = question_case['p']
        pairs = question_case['pairs']
        t = question_case['t']
        
        input_lines = [
            f"{n} {m}",
            ' '.join(map(str, p))
        ]
        input_lines.extend(f"{a} {b}" for a, b in pairs)
        input_lines.append(str(t))
        input_str = '\n'.join(input_lines)
        
        prompt = f"""你正在参与一个英雄选择游戏，请根据以下规则和输入数据做出最优决策：

**游戏规则**
- 两队各有 {n} 名玩家，共有 {2*n} 个英雄。
- 英雄战斗力依次为：{', '.join(map(str, p))}。
- 存在 {m} 对特殊组合：{'；'.join(f'英雄{a}与英雄{b}' for a, b in pairs)}。当一方选择组合中的一个英雄，对方下回合必须选择另一个。
- 由你控制的队伍先选（t=1）或后选（t=2），当前 t={t}。

**目标**
通过最优策略选择英雄，确保无论对方如何选择，总战斗力最大化。

**输入格式**
{input_str}

**输出要求**
将你的选择按顺序放入[answer]标签，例如：
[answer]
6
5
3
[/answer]

请严格按照规则输出选择顺序："""
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        solution = []
        for line in last_answer.split('\n'):
            line = line.strip()
            if line.isdigit():
                solution.append(int(line))
        return solution if solution else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != identity['n']:
            return False
        
        try:
            user_choices = [x-1 for x in solution]
        except:
            return False
        
        n = identity['n']
        p = identity['p']
        pairs = identity['pairs']
        t = identity['t']
        jury_priority = [x-1 for x in identity['jury_priority']]
        opp = {}
        for a, b in pairs:
            a_idx, b_idx = a-1, b-1
            opp[a_idx] = b_idx
            opp[b_idx] = a_idx
        
        available = set(range(2*n))
        user_team = []
        la = -1
        turn = t
        jury_ptr = 0
        
        for x in user_choices:
            if turn != 1:
                return False
            if x not in available:
                return False
            if la != -1 and la in opp and opp[la] in available and x != opp[la]:
                return False
            user_team.append(x)
            available.remove(x)
            la = x
            turn = 2
            
            while turn == 2 and available:
                if la in opp and opp[la] in available:
                    jury_choice = opp[la]
                else:
                    for hero in jury_priority:
                        if hero in available:
                            jury_choice = hero
                            break
                available.remove(jury_choice)
                la = jury_choice
                turn = 1
        
        total = sum(p[i] for i in user_team)
        return total == identity['correct_total']
