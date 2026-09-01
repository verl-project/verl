"""# 

### 谜题描述
Sereja has two sequences a1, a2, ..., an and b1, b2, ..., bm, consisting of integers. One day Sereja got bored and he decided two play with them. The rules of the game was very simple. Sereja makes several moves, in one move he can perform one of the following actions:

  1. Choose several (at least one) first elements of sequence a (non-empty prefix of a), choose several (at least one) first elements of sequence b (non-empty prefix of b); the element of sequence a with the maximum index among the chosen ones must be equal to the element of sequence b with the maximum index among the chosen ones; remove the chosen elements from the sequences. 
  2. Remove all elements of both sequences. 



The first action is worth e energy units and adds one dollar to Sereja's electronic account. The second action is worth the number of energy units equal to the number of elements Sereja removed from the sequences before performing this action. After Sereja performed the second action, he gets all the money that he earned on his electronic account during the game.

Initially Sereja has s energy units and no money on his account. What maximum number of money can Sereja get? Note, the amount of Seraja's energy mustn't be negative at any time moment.

Input

The first line contains integers n, m, s, e (1 ≤ n, m ≤ 105; 1 ≤ s ≤ 3·105; 103 ≤ e ≤ 104). The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 105). The third line contains m integers b1, b2, ..., bm (1 ≤ bi ≤ 105).

Output

Print a single integer — maximum number of money in dollars that Sereja can get.

Examples

Input

5 5 100000 1000
1 2 3 4 5
3 2 4 5 1


Output

3


Input

3 4 3006 1000
1 2 3
1 2 4 3


Output

2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int INF = 1E9 + 7;
template <class C>
void mini(C& a4, C b4) {
  a4 = in(a4, b4);
}
set<int> gdzie[2][100005];
int t[2][100005];
int best[100005][305];
int main() {
  ios_base::sync_with_stdio(false);
  int n, m, s, e;
  cin >> n >> m >> s >> e;
  for (int i = (1); i <= (n); ++i) {
    cin >> t[0][i];
  }
  for (int i = (1); i <= (m); ++i) {
    cin >> t[1][i];
    gdzie[1][t[1][i]].insert(i);
  }
  for (int i = (1); i <= (n); ++i) {
    int a = t[0][i];
    if (((int)(gdzie[1][a]).size())) {
      int pos = *(gdzie[1][a].begin());
      best[i][1] = pos;
    } else
      best[i][1] = INF;
  }
  for (int k = (2); k <= (300); ++k) {
    int be = INF;
    for (int i = (1); i <= (n); ++i) {
      best[i][k] = INF;
      int a = t[0][i];
      set<int>::iterator l = gdzie[1][a].upper_bound(be);
      if (l != gdzie[1][a].end()) best[i][k] = *l;
      be = min(be, best[i][k - 1]);
    }
  }
  int wyn = 0;
  for (int k = (1); k <= (300); ++k) {
    int ene_cost = INF;
    for (int i = (1); i <= (n); ++i) {
      ene_cost = min(ene_cost, i + best[i][k]);
    }
    ene_cost += k * e;
    if (ene_cost <= s) wyn = k;
  }
  cout << wyn << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cserejaandtwosequencesbootcamp(Basebootcamp):
    def __init__(self, k=3, n=5, m=5, e=1000):
        self.k = k  # Expected correct answer (maximum number of moves)
        self.n = n  # Length of sequence a
        self.m = m  # Length of sequence b
        self.e = e  # Energy cost per move of type 1

    def case_generator(self):
        # Generate k unique common elements
        common = list(range(1, self.k + 1))
        
        # Build sequence a with the common elements followed by unique elements
        a = common.copy()
        used_a = set(common)
        remaining_a = []
        while len(remaining_a) < self.n - self.k:
            num = random.randint(self.k + 1, 100000)
            if num not in used_a:
                remaining_a.append(num)
                used_a.add(num)
        a += remaining_a[:self.n - self.k]
        
        # Build sequence b with the same common elements followed by unique elements not overlapping with a's remaining
        b = common.copy()
        used_b = set(common)
        remaining_b = []
        while len(remaining_b) < self.m - self.k:
            num = random.randint(self.k + 10001, 200000)  # Ensure no overlap with a's remaining
            if num not in used_b:
                remaining_b.append(num)
                used_b.add(num)
        b += remaining_b[:self.m - self.k]
        
        # Calculate s to ensure it's exactly enough for k moves plus the remaining elements cost
        s = self.k * self.e + (self.n + self.m - 2 * self.k)
        
        return {
            'n': self.n,
            'm': self.m,
            's': s,
            'e': self.e,
            'a': a,
            'b': b,
            'correct_k': self.k
        }
    
    @staticmethod
    def prompt_func(question_case):
        a_str = ' '.join(map(str, question_case['a']))
        b_str = ' '.join(map(str, question_case['b']))
        prompt = (
            "Sereja has two sequences of integers and can perform two types of operations to earn dollars. Each move of type 1 costs a fixed energy (e) and earns $1. Move type 2 removes all remaining elements, costing energy equal to the number of elements left. Determine the maximum dollars Sereja can earn without his energy dropping below zero.\n\n"
            "Input format:\n"
            "- First line: n m s e (sequence lengths, initial energy, energy cost per move type 1)\n"
            "- Second line: a_1 a_2 ... a_n\n"
            "- Third line: b_1 b_2 ... b_m\n\n"
            f"Input:\n{question_case['n']} {question_case['m']} {question_case['s']} {question_case['e']}\n"
            f"{a_str}\n"
            f"{b_str}\n\n"
            "Output the maximum dollars as an integer enclosed within [answer] and [/answer], e.g., [answer]3[/answer].\n"
            "Output:\n"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_k']
