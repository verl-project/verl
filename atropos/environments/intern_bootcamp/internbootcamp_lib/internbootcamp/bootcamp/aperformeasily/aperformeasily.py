"""# 

### 谜题描述
After battling Shikamaru, Tayuya decided that her flute is too predictable, and replaced it with a guitar. The guitar has 6 strings and an infinite number of frets numbered from 1. Fretting the fret number j on the i-th string produces the note a_{i} + j.

Tayuya wants to play a melody of n notes. Each note can be played on different string-fret combination. The easiness of performance depends on the difference between the maximal and the minimal indices of used frets. The less this difference is, the easier it is to perform the technique. Please determine the minimal possible difference.

For example, if a = [1, 1, 2, 2, 3, 3], and the sequence of notes is 4, 11, 11, 12, 12, 13, 13 (corresponding to the second example), we can play the first note on the first string, and all the other notes on the sixth string. Then the maximal fret will be 10, the minimal one will be 3, and the answer is 10 - 3 = 7, as shown on the picture.

<image>

Input

The first line contains 6 space-separated numbers a_{1}, a_{2}, ..., a_{6} (1 ≤ a_{i} ≤ 10^{9}) which describe the Tayuya's strings.

The second line contains the only integer n (1 ≤ n ≤ 100 000) standing for the number of notes in the melody.

The third line consists of n integers b_{1}, b_{2}, ..., b_{n} (1 ≤ b_{i} ≤ 10^{9}), separated by space. They describe the notes to be played. It's guaranteed that b_i > a_j for all 1≤ i≤ n and 1≤ j≤ 6, in other words, you can play each note on any string.

Output

Print the minimal possible difference of the maximal and the minimal indices of used frets.

Examples

Input


1 4 100 10 30 5
6
101 104 105 110 130 200


Output


0


Input


1 1 2 2 3 3
7
13 4 11 12 11 13 12


Output


7

Note

In the first sample test it is optimal to play the first note on the first string, the second note on the second string, the third note on the sixth string, the fourth note on the fourth string, the fifth note on the fifth string, and the sixth note on the third string. In this case the 100-th fret is used each time, so the difference is 100 - 100 = 0.

<image>

In the second test it's optimal, for example, to play the second note on the first string, and all the other notes on the sixth string. Then the maximal fret will be 10, the minimal one will be 3, and the answer is 10 - 3 = 7.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int L = 6;
const int N = 1e5 + 5;
int A[L + 1];
int licz[N];
int main() {
  for (int i = 1; i <= L; i++) scanf(\"%d\", &A[i]);
  int n;
  scanf(\"%d\", &n);
  vector<pair<int, int>> broom;
  for (int i = 1; i <= n; i++) {
    int x;
    scanf(\"%d\", &x);
    for (int j = 1; j <= L; j++) {
      if (x > A[j]) broom.push_back({x - A[j], i});
    }
  }
  sort(broom.begin(), broom.end());
  int not_covered = n;
  int i = 0;
  int j = -1;
  while (not_covered > 0) {
    j++;
    if (licz[broom[j].second] == 0) not_covered--;
    licz[broom[j].second]++;
  }
  int res = broom[j].first - broom[i].first;
  while (i < int(broom.size())) {
    licz[broom[i].second]--;
    while (j < int(broom.size()) - 1 && licz[broom[i].second] == 0) {
      j++;
      licz[broom[j].second]++;
    }
    if (licz[broom[i].second] == 0) break;
    i++;
    res = min(res, broom[j].first - broom[i].first);
  }
  printf(\"%d\n\", res);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Aperformeasilybootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'a_min': params.get('a_min', 1),
            'a_max': params.get('a_max', 1000),
            'b_min': params.get('b_min', 1),
            'b_max': params.get('b_max', 100),
            'n_min': params.get('n_min', 1),
            'n_max': params.get('n_max', 100),
        }
        # 参数校验
        assert self.params['a_min'] <= self.params['a_max'], "Invalid a range"
        assert self.params['b_min'] >= 1, "b_min should be ≥1"

    def case_generator(self):
        # 生成6个随机a值，确保至少有一个最大值
        max_a = random.randint(self.params['a_min'], self.params['a_max'])
        a = [max_a] + [random.randint(self.params['a_min'], max_a) for _ in range(5)]
        random.shuffle(a)
        
        # 生成b序列（保证所有b_i > a_j）
        n = random.randint(self.params['n_min'], self.params['n_max'])
        min_fret = max_a + self.params['b_min']
        b = [random.randint(min_fret, max_a + self.params['b_max']) for _ in range(n)]
        
        return {'a': a, 'b': b}

    @staticmethod
    def prompt_func(question_case):
        a_str = ' '.join(map(str, sorted(question_case['a'], reverse=True)))
        b_str = ' '.join(map(str, question_case['b']))
        return (
            f"Tayuya的吉他有6根弦，基准音符为：{a_str}\n"
            f"需要弹奏{len(question_case['b'])}个音符：{b_str}\n"
            "请计算使用品位的极差最小值。答案格式：[answer]数值[/answer]"
        )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(float(matches[-1].strip().replace(',','')))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == compute_min_difference(identity['a'], identity['b'])
        except:
            return False

def compute_min_difference(a_list, b_list):
    events = []
    for note_idx, b in enumerate(b_list):
        for a in a_list:
            if b > a:  # 题目保证条件，实际可省略
                events.append((b - a, note_idx))
    events.sort()
    
    freq = defaultdict(int)
    n = len(b_list)
    l = 0
    min_diff = float('inf')
    unique = 0
    
    for r in range(len(events)):
        # 扩展右边界
        rf, rn = events[r]
        if freq[rn] == 0:
            unique += 1
        freq[rn] += 1
        
        # 收缩左边界
        while unique == n and l <= r:
            current_diff = rf - events[l][0]
            if current_diff < min_diff:
                min_diff = current_diff
            lf, ln = events[l]
            freq[ln] -= 1
            if freq[ln] == 0:
                unique -= 1
            l += 1
            
    return min_diff
