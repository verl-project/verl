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
int N, K;
struct Node {
  int mx[10], mn[10];
  int sz;
  bool operator<(const Node &b) const {
    for (int i = 0; i < K; i++) {
      if (mn[i] < b.mx[i]) {
        return true;
      }
    }
    return false;
  }
  bool operator>(const Node &b) const {
    for (int i = 0; i < K; i++) {
      if (mx[i] > b.mn[i]) {
        return true;
      }
    }
    return false;
  }
};
int main() {
  cin >> N >> K;
  set<Node> st;
  for (int i = 0; i < N; i++) {
    Node tmp;
    tmp.sz = 1;
    for (int j = 0; j < K; j++) {
      cin >> tmp.mn[j];
      tmp.mx[j] = tmp.mn[j];
    }
    for (auto it = st.lower_bound(tmp); it != st.begin();
         it = st.lower_bound(tmp)) {
      it--;
      if ((*it) > tmp) {
        for (int j = 0; j < K; j++) {
          tmp.mn[j] = min(tmp.mn[j], it->mn[j]);
          tmp.mx[j] = max(tmp.mx[j], it->mx[j]);
        }
        tmp.sz += it->sz;
        st.erase(it);
      } else {
        break;
      }
    }
    st.insert(tmp);
    cout << (--st.end())->sz << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import bisect
import random
from typing import List, Optional

from bootcamp import Basebootcamp

class Node:
    def __init__(self, s_list):
        self.mx = s_list.copy()
        self.mn = s_list.copy()
        self.sz = 1
    
    def __lt__(self, other: 'Node') -> bool:
        for i in range(len(self.mn)):
            if self.mn[i] < other.mx[i]:
                return True
        return False
    
    def is_greater_than(self, other: 'Node') -> bool:
        for i in range(len(self.mx)):
            if self.mx[i] > other.mn[i]:
                return True
        return False

class Etournamentbootcamp(Basebootcamp):
    def __init__(self, max_n=100, max_k=10):
        self.max_n = max_n
        self.max_k = max_k
    
    def case_generator(self) -> dict:
        n = random.randint(1, self.max_n)
        k = random.randint(1, min(self.max_k, 10))
        
        # Generate unique values for each sport
        athletes = []
        sport_values = []
        for j in range(k):
            values = random.sample(range(1, 10**9), n)
            sport_values.append(values)
        
        # Transpose to get athletes' stats
        athletes = [[sport_values[j][i] for j in range(k)] for i in range(n)]
        
        # Compute correct output
        correct_output = self._compute_correct_output(n, k, athletes)
        
        return {
            'n': n,
            'k': k,
            'athletes': athletes,
            'correct_output': correct_output
        }
    
    def _compute_correct_output(self, n: int, k: int, athletes: List[List[int]]) -> List[int]:
        nodes = []
        correct_output = []
        for s in athletes:
            tmp = Node(s)
            while True:
                pos = bisect.bisect_left(nodes, tmp)
                merged = False
                while pos > 0:
                    pos -= 1
                    current_node = nodes[pos]
                    if current_node.is_greater_than(tmp):
                        tmp.sz += current_node.sz
                        for j in range(k):
                            tmp.mn[j] = min(tmp.mn[j], current_node.mn[j])
                            tmp.mx[j] = max(tmp.mx[j], current_node.mx[j])
                        del nodes[pos]
                        merged = True
                        break
                    else:
                        break
                if not merged:
                    break
            bisect.insort(nodes, tmp)
            correct_output.append(nodes[-1].sz if nodes else 0)
        return correct_output
    
    @staticmethod
    def prompt_func(question_case) -> str:
        prompt = """你正在参加一个包含多种运动的锦标赛分析任务。请根据以下比赛数据，计算每年锦标赛可能的冠军人数。

输入格式：
第一行包含两个整数n和k，表示锦标赛的年数和运动种类数。
接下来n行，每行包含k个整数，表示第i年加入的运动员在各个运动中的能力值（保证同一运动的能力值唯一）。

输出格式：
输出n行，每行一个整数，表示对应年份可能的冠军人数。

题目数据：
n = {n}
k = {k}
运动员能力值：
{athletes}

请将你的答案放置在[answer]标签内，例如：
[answer]
1
2
3
[/answer]

你需要确保：
1. 严格按照输入数据计算正确结果
2. 输出格式与要求的完全一致
3. 将最终答案放在[answer]标签内""".format(
            n=question_case['n'],
            k=question_case['k'],
            athletes='\n'.join(' '.join(map(str, row)) for row in question_case['athletes'])
        )
        return prompt
    
    @staticmethod
    def extract_output(output: str) -> Optional[List[int]]:
        start_tag = '[answer]'
        end_tag = '[/answer]'
        start_idx = output.rfind(start_tag)
        end_idx = output.rfind(end_tag)
        
        if start_idx == -1 or end_idx == -1:
            return None
        
        answer_lines = output[start_idx+len(start_tag):end_idx].strip().split('\n')
        try:
            return [int(line.strip()) for line in answer_lines if line.strip()]
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity) -> bool:
        return solution == identity['correct_output']
