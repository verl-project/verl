"""# 

### 谜题描述
Recently a tournament in k kinds of sports has begun in Berland. Vasya wants to make money on the bets.

The scheme of the tournament is very mysterious and not fully disclosed. Competitions are held back to back, each of them involves two sportsmen who have not left the tournament yet. Each match can be held in any of the k kinds of sport. Loser leaves the tournament. The last remaining sportsman becomes the winner. Apart of this, the scheme can be arbitrary, it is not disclosed in advance.

Vasya knows powers of sportsmen in each kind of sport. He believes that the sportsmen with higher power always wins.

The tournament is held every year, and each year one new participant joins it. In the first tournament, only one sportsman has participated, in the second there were two sportsmen, and so on. Vasya has been watching the tournament for the last n years. Help him to find the number of possible winners for each of the n tournaments.

Input

The first line contains two integers n and k (1 ≤ n ≤ 5·104, 1 ≤ k ≤ 10) — the number of tournaments and the number of kinds of sport, respectively.

Each of the next n lines contains k integers si1, si2, ..., sik (1 ≤ sij ≤ 109), where sij is the power of the i-th sportsman in the j-th kind of sport. The sportsman with higher powers always wins. It's guaranteed that for any kind of sport all of these powers are distinct.

Output

For each of the n tournaments output the number of contenders who can win.

Examples

Input

3 2
1 5
5 1
10 10


Output

1
2
1


Input

3 2
2 2
3 3
1 10


Output

1
1
3


Input

3 2
2 3
1 1
3 2


Output

1
1
2

Note

In the first sample:

In the first tournament there is only one sportsman, and he is the winner.

In the second tournament, there are two sportsmen, and everyone can defeat another, depending on kind of sports.

In the third tournament, the third sportsman in the strongest in both kinds of sports, so he is the winner regardless of the scheme.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 10;
int n, k;
struct Node {
  int mx[N], mn[N], siz;
  bool operator<(const Node &a) const {
    for (int i = 0; i < k; ++i)
      if (mx[i] > a.mn[i]) return 0;
    return 1;
  }
} t;
set<Node> s;
set<Node>::iterator it;
int main() {
  scanf(\"%d%d\", &n, &k);
  for (int i = 1; i <= n; ++i) {
    t.siz = 1;
    for (int j = 0; j < k; ++j) scanf(\"%d\", &t.mx[j]), t.mn[j] = t.mx[j];
    while ((it = s.find(t)) != s.end()) {
      t.siz += it->siz;
      for (int j = 0; j < k; ++j)
        t.mx[j] = max(t.mx[j], it->mx[j]), t.mn[j] = min(t.mn[j], it->mn[j]);
      s.erase(it);
    }
    s.insert(t);
    printf(\"%d\n\", (--s.end())->siz);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bisect import bisect_left, insort
from bootcamp import Basebootcamp

class Ctournamentbootcamp(Basebootcamp):
    def __init__(self, n=3, k=2):
        self.n = n
        self.k = k
    
    def case_generator(self):
        n, k = self.n, self.k
        sportsmen = []
        # Generate unique values for each sport
        for _ in range(k):
            base = random.randint(1, 10**9)
            step = random.choice([1, 10, 100])
            values = [base + i*step for i in random.sample(range(10**5), n)]
            sportsmen.append(values)
        
        # Transpose to get player stats
        players = list(zip(*sportsmen))
        
        return {
            'n': n,
            'k': k,
            'sportsmen': players,
            'answers': self.calculate_answers(players, k)
        }
    
    @staticmethod
    def calculate_answers(players, k):
        class Node:
            __slots__ = ['mx', 'mn', 'siz']
            def __init__(self, stats):
                self.mx = list(stats)
                self.mn = list(stats)
                self.siz = 1
            def __lt__(self, other):
                return all(a <= b for a, b in zip(self.mx, other.mn))
        
        nodes = []
        answers = []
        for stats in players:
            current = Node(stats)
            while True:
                # Find merge candidates using bisect
                idx = bisect_left(nodes, current)
                merged = False
                
                # Check left neighbor
                if idx > 0 and current < nodes[idx-1]:
                    candidate = nodes.pop(idx-1)
                    current.siz += candidate.siz
                    current.mx = [max(a,b) for a,b in zip(current.mx, candidate.mx)]
                    current.mn = [min(a,b) for a,b in zip(current.mn, candidate.mn)]
                    merged = True
                
                # Check right neighbor
                if idx < len(nodes) and current < nodes[idx]:
                    candidate = nodes.pop(idx)
                    current.siz += candidate.siz
                    current.mx = [max(a,b) for a,b in zip(current.mx, candidate.mx)]
                    current.mn = [min(a,b) for a,b in zip(current.mn, candidate.mn)]
                    merged = True
                
                if not merged: break
            
            insort(nodes, current)
            answers.append(nodes[-1].siz)
        return answers
    
    @staticmethod
    def prompt_func(case):
        prompt = [
            "体育锦标赛分析任务：",
            f"共有 {case['n']} 届锦标赛，每届新增1名选手，需计算每届可能的冠军数量。",
            "规则要点：",
            "1. 每场比赛可任选运动类型进行较量",
            "2. 高能力值选手必胜低能力值选手",
            "3. 最后剩下的选手获胜",
            "选手能力矩阵："
        ]
        for i, stats in enumerate(case['sportsmen']):
            prompt.append(f"第{i+1}年选手: {' '.join(map(str, stats))}")
        
        prompt.append("请输出每届可能的冠军数量，格式：\n[answer]1 2 3[/answer]")
        return '\n'.join(prompt)
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*((?:\d+\s*)+)\[/answer\]', output)
        if not matches: return None
        try:
            return list(map(int, matches[-1].strip().split()))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answers']
