"""# 

### 谜题描述
Ujan has a lot of numbers in his boxes. He likes order and balance, so he decided to reorder the numbers.

There are k boxes numbered from 1 to k. The i-th box contains n_i integer numbers. The integers can be negative. All of the integers are distinct.

Ujan is lazy, so he will do the following reordering of the numbers exactly once. He will pick a single integer from each of the boxes, k integers in total. Then he will insert the chosen numbers — one integer in each of the boxes, so that the number of integers in each box is the same as in the beginning. Note that he may also insert an integer he picked from a box back into the same box.

Ujan will be happy if the sum of the integers in each box is the same. Can he achieve this and make the boxes perfectly balanced, like all things should be?

Input

The first line contains a single integer k (1 ≤ k ≤ 15), the number of boxes. 

The i-th of the next k lines first contains a single integer n_i (1 ≤ n_i ≤ 5 000), the number of integers in box i. Then the same line contains n_i integers a_{i,1}, …, a_{i,n_i} (|a_{i,j}| ≤ 10^9), the integers in the i-th box. 

It is guaranteed that all a_{i,j} are distinct.

Output

If Ujan cannot achieve his goal, output \"No\" in a single line. Otherwise in the first line output \"Yes\", and then output k lines. The i-th of these lines should contain two integers c_i and p_i. This means that Ujan should pick the integer c_i from the i-th box and place it in the p_i-th box afterwards.

If there are multiple solutions, output any of those.

You can print each letter in any case (upper or lower).

Examples

Input


4
3 1 7 4
2 3 2
2 8 5
1 10


Output


Yes
7 2
2 3
5 1
10 4


Input


2
2 3 -2
2 -1 5


Output


No


Input


2
2 -10 10
2 0 -20


Output


Yes
-10 2
-20 1

Note

In the first sample, Ujan can put the number 7 in the 2nd box, the number 2 in the 3rd box, the number 5 in the 1st box and keep the number 10 in the same 4th box. Then the boxes will contain numbers \{1,5,4\}, \{3, 7\}, \{8,2\} and \{10\}. The sum in each box then is equal to 10.

In the second sample, it is not possible to pick and redistribute the numbers in the required way.

In the third sample, one can swap the numbers -20 and -10, making the sum in each box equal to -10.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 16;
const int M = 5010;
const int inf = 0x3f3f3f3f;
int a[N][M], c[N], two[N << 1], f[1 << N], mix[1 << N];
long long sum[N], tot;
map<long long, int> mp;
pair<int, int> loop[1 << N], ans[N];
void init(void) {
  two[0] = 1;
  for (int i = 1; i <= 20; ++i) {
    two[i] = two[i - 1] << 1;
  }
}
void dfs(int sta) {
  if (mix[sta] != inf) {
    dfs(mix[sta]);
    dfs(sta ^ mix[sta]);
  } else {
    int x = loop[sta].first, y = loop[sta].second;
    int want = tot - sum[x] + y;
    while (want != y) {
      ans[mp[want]].first = want;
      ans[mp[want]].second = x;
      x = mp[want];
      want += tot - sum[mp[want]];
    }
    ans[mp[want]].first = want;
    ans[mp[want]].second = x;
  }
}
int main(void) {
  ios::sync_with_stdio(false), cin.tie(0);
  init();
  int k;
  cin >> k;
  for (int i = 1; i <= k; ++i) {
    cin >> c[i];
    for (int j = 1; j <= c[i]; ++j) {
      cin >> a[i][j];
      sum[i] += a[i][j];
      mp[a[i][j]] = i;
    }
    tot += sum[i];
  }
  if (tot % k) {
    cout << \"No\" << endl;
    return 0;
  }
  tot /= k;
  for (int i = 1; i <= k; ++i) {
    for (int j = 1; j <= c[i]; ++j) {
      int sta = 0;
      sta |= two[i - 1];
      long long want = tot - sum[i] + a[i][j];
      while (mp.count(want) && !(sta & two[mp[want] - 1])) {
        sta |= two[mp[want] - 1], want += tot - sum[mp[want]];
      }
      if (want == a[i][j]) {
        f[sta] = 1;
        loop[sta].first = i, loop[sta].second = a[i][j];
      }
    }
  }
  memset(mix, inf, sizeof mix);
  for (int msk = 1; msk < two[k]; ++msk) {
    for (int i = msk; i > 0; i = (i - 1) & msk) {
      if (!f[msk] && f[i] && f[msk ^ i]) {
        f[msk] = 1;
        mix[msk] = i;
      }
    }
  }
  if (!f[two[k] - 1]) {
    cout << \"No\" << endl;
    return 0;
  }
  dfs(two[k] - 1);
  cout << \"Yes\" << endl;
  for (int i = 1; i <= k; ++i) {
    cout << ans[i].first << ' ' << ans[i].second << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict

class Esumbalancebootcamp(Basebootcamp):
    def __init__(self, k=4):
        self.k = k
    
    def case_generator(self):
        for _ in range(100):
            try:
                boxes = list(range(1, self.k+1))
                perm = random.sample(boxes, self.k)
                q = {}
                for j in range(self.k):
                    target_box = perm[j]
                    q[target_box] = j+1

                existing = set()
                c = []
                for _ in range(self.k):
                    while True:
                        num = random.randint(-100, 100)
                        if num not in existing:
                            existing.add(num)
                            c.append(num)
                            break
                
                found = False
                for _ in range(100):
                    target = random.randint(-100, 100)
                    all_x = set()
                    valid = True
                    for i in range(1, self.k+1):
                        j = q.get(i, None)
                        if j is None:
                            valid = False
                            break
                        x_i = target - c[j-1]
                        if x_i in existing or x_i == c[i-1] or x_i in all_x:
                            valid = False
                            break
                        all_x.add(x_i)
                    if valid:
                        found = True
                        break
                if not found:
                    continue

                boxes_data = []
                all_numbers = set(c)
                all_x = set()
                for i in range(1, self.k+1):
                    j = q[i]
                    x_i = target - c[j-1]
                    all_x.add(x_i)
                    box_numbers = [c[i-1], x_i]
                    random.shuffle(box_numbers)
                    boxes_data.append({
                        'n_i': len(box_numbers),
                        'numbers': box_numbers
                    })
                all_numbers.update(all_x)
                if len(all_numbers) != 2*self.k:
                    continue

                case = {
                    'k': self.k,
                    'boxes': boxes_data,
                    'target': target,
                }
                return case
            except:
                continue
        
        return {
            'k': 4,
            'boxes': [
                {'n_i': 3, 'numbers': [1, 7, 4]},
                {'n_i': 2, 'numbers': [3, 2]},
                {'n_i': 2, 'numbers': [8, 5]},
                {'n_i': 1, 'numbers': [10]}
            ],
            'target': 10
        }
    
    @staticmethod
    def prompt_func(question_case):
        problem = "Ujan has reordered his boxes of numbers. Each box contains distinct integers. He picks one number from each box and redistributes them into the boxes (each box receives exactly one number) so that the sum of the numbers in each box becomes the same. Your task is to determine if this is possible.\n\n"
        problem += f"There are {question_case['k']} boxes:\n"
        for i, box in enumerate(question_case['boxes'], 1):
            nums = ' '.join(map(str, box['numbers']))
            problem += f"Box {i}: {box['n_i']} numbers: {nums}\n"
        problem += "\nIf possible, output 'Yes' followed by the chosen numbers and their destination boxes. Each line after 'Yes' should contain two integers: the number selected from a box and the box it's placed into. Each box must receive exactly one number. If not possible, output 'No'.\n"
        problem += "Enclose your final answer within [answer] and [/answer], like this:\n[answer]\nYes\n7 2\n2 3\n5 1\n10 4\n[/answer]"
        return problem
    
    @staticmethod
    def extract_output(output):
        answer_pattern = re.compile(r'\[answer\](.*?)\[/answer\]', re.DOTALL)
        matches = answer_pattern.findall(output)
        if not matches:
            return None
        content = matches[-1].strip()
        if 'No' in content.split('\n')[0].strip().lower():
            return 'No'
        lines = [line.strip() for line in content.split('\n')]
        if len(lines) < 1 or lines[0].lower() != 'yes':
            return None
        solution = []
        for line in lines[1:]:
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                return None
            try:
                c = int(parts[0])
                p = int(parts[1])
                solution.append((c, p))
            except:
                return None
        return solution if len(solution) == len(lines)-1 else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == 'No':
            return False
        k = identity['k']
        boxes = identity['boxes']
        target = identity['target']
        if len(solution) != k:
            return False
        
        selected = set()
        box_numbers = [set(box['numbers']) for box in boxes]
        for i in range(k):
            c, p = solution[i]
            if p < 1 or p > k or c not in box_numbers[i] or c in selected:
                return False
            selected.add(c)
        
        outgoing = [0] * (k+1)
        incoming = [0] * (k+1)
        for i, (c, p) in enumerate(solution):
            outgoing[i+1] += c
            incoming[p] += c
        
        for i in range(k):
            original_sum = sum(boxes[i]['numbers'])
            new_sum = original_sum - outgoing[i+1] + incoming[i+1]
            if new_sum != target:
                return False
        
        return True
