"""# 

### 谜题描述
A very tense moment: n cowboys stand in a circle and each one points his colt at a neighbor. Each cowboy can point the colt to the person who follows or precedes him in clockwise direction. Human life is worthless, just like in any real western.

The picture changes each second! Every second the cowboys analyse the situation and, if a pair of cowboys realize that they aim at each other, they turn around. In a second all such pairs of neighboring cowboys aiming at each other turn around. All actions happen instantaneously and simultaneously in a second.

We'll use character \"A\" to denote a cowboy who aims at his neighbour in the clockwise direction, and character \"B\" for a cowboy who aims at his neighbour in the counter clockwise direction. Then a string of letters \"A\" and \"B\" will denote the circle of cowboys, the record is made from the first of them in a clockwise direction.

For example, a circle that looks like \"ABBBABBBA\" after a second transforms into \"BABBBABBA\" and a circle that looks like \"BABBA\" transforms into \"ABABB\".

<image> This picture illustrates how the circle \"BABBA\" transforms into \"ABABB\" 

A second passed and now the cowboys' position is described by string s. Your task is to determine the number of possible states that lead to s in a second. Two states are considered distinct if there is a cowboy who aims at his clockwise neighbor in one state and at his counter clockwise neighbor in the other state.

Input

The input data consists of a single string s. Its length is from 3 to 100 characters, inclusive. Line s consists of letters \"A\" and \"B\".

Output

Print the sought number of states.

Examples

Input

BABBBABBA


Output

2


Input

ABABB


Output

2


Input

ABABAB


Output

4

Note

In the first sample the possible initial states are \"ABBBABBAB\" and \"ABBBABBBA\".

In the second sample the possible initial states are \"AABBB\" and \"BABBA\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MOD = 1000000007;
template <class T>
typename T::value_type arr_sum(const T& v, int n) {
  typename T::value_type sum = 0;
  for (int i = (0); i < (n); ++i) sum += v[i];
  return sum;
}
struct Sync_stdio {
  Sync_stdio() {
    cin.tie(NULL);
    ios_base::sync_with_stdio(false);
  }
} _sync_stdio;
int main() {
  string s;
  cin >> s;
  int n = s.size();
  const int C = 53;
  vector<vector<long long>> dp(C, vector<long long>(2));
  dp[0][0] = 1;
  dp[0][1] = 0;
  for (int i = (1); i < (C); ++i) {
    dp[i][1] = dp[i - 1][0];
    dp[i][0] = dp[i - 1][0] + dp[i - 1][1];
  }
  if (s.size() % 2 == 0) {
    string t;
    string t2;
    for (int i = (0); i < (n); ++i) {
      t += 'A' + i % 2;
      t2 += 'B' - i % 2;
    }
    if (s == t || s == t2) {
      cout << dp[n / 2][0] + dp[n / 2 - 1][1];
      return 0;
    }
  }
  while (s.back() != s.front()) {
    s.insert(s.begin(), s.back());
    s.pop_back();
  }
  long long res = 1;
  for (int i = (0); i < (n - 1); ++i) {
    if (s[i] == s[i + 1]) {
      continue;
    }
    int j = i;
    while (j < n - 1 && s[j] != s[j + 1]) {
      ++j;
    }
    int t = j - i + 1;
    if (s[j] == 'B') {
      --t;
    }
    if (s[i] == 'A') {
      --t;
    }
    t /= 2;
    if (t == 0 && s[i] == 'A' && s[j] == 'B') {
      cout << 0;
      return 0;
    }
    if (s[j] == 'B') {
      --t;
    }
    if (s[i] == 'A') {
      --t;
    }
    t = max(t, 0);
    res *= dp[t][0] + dp[t][1];
    i = j;
  }
  cout << res;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from random import Random
from bootcamp import Basebootcamp

class Ccowboysbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.params = params
        self.min_length = params.get('min_length', 3)
        self.max_length = params.get('max_length', 100)
        self.random_seed = params.get('random_seed', None)
        self.rng = Random(self.random_seed)
        
        # Precompute dp表用于状态计算
        self.dp = [[0]*2 for _ in range(53)]
        self.dp[0][0] = 1
        self.dp[0][1] = 0
        for i in range(1, 53):
            self.dp[i][1] = self.dp[i-1][0]
            self.dp[i][0] = self.dp[i-1][0] + self.dp[i-1][1]

    @staticmethod
    def transform(s_prev):
        n = len(s_prev)
        new_chars = list(s_prev)
        swap_indices = []
        for i in range(n):
            j = (i+1) % n
            if s_prev[i] == 'A' and s_prev[j] == 'B':
                swap_indices.append(i)
        for i in swap_indices:
            j = (i+1) % n
            new_chars[i], new_chars[j] = new_chars[j], new_chars[i]
        return ''.join(new_chars)

    def _calculate_answer(self, s):
        n = len(s)
        if n % 2 == 0:
            pattern1 = ''.join('A' if i%2 ==0 else 'B' for i in range(n))
            pattern2 = ''.join('B' if i%2 ==0 else 'A' for i in range(n))
            if s in (pattern1, pattern2):
                return self.dp[n//2][0] + self.dp[n//2-1][1]
        
        s_list = list(s)
        while s_list[-1] != s_list[0]:
            s_list.insert(0, s_list.pop())
        
        result = 1
        i = 0
        n = len(s_list)
        while i < n-1:
            if s_list[i] == s_list[i+1]:
                i += 1
                continue
            
            j = i
            while j < n-1 and s_list[j] != s_list[j+1]:
                j += 1
            
            segment_length = j - i + 1
            if s_list[j] == 'B':
                segment_length -= 1
            if s_list[i] == 'A':
                segment_length -= 1
            segment_length = max(segment_length // 2, 0)
            
            if segment_length == 0 and s_list[i] == 'A' and s_list[j] == 'B':
                return 0
            
            result *= (self.dp[segment_length][0] + self.dp[segment_length][1])
            i = j
        
        return result

    def case_generator(self):
        for _ in range(1000):
            length = self.rng.randint(self.min_length, self.max_length)
            prev_state = ''.join(self.rng.choice(['A','B']) for _ in range(length))
            current_state = self.transform(prev_state)
            answer = self._calculate_answer(current_state)
            if answer > 0:
                return {'problem': current_state, 'answer': answer}
        return {'problem': 'ABBBABBBA', 'answer': 2}

    @staticmethod
    def prompt_func(question_case):
        s = question_case['problem']
        n = len(s)  # 修正的关键点：显式计算字符串长度
        return f"""## 牛仔决斗状态转换问题

你正在观察一个由{n}个牛仔组成的环形决斗场。每个牛仔用'A'表示指向顺时针方向的下一个牛仔，'B'表示指向逆时针方向。每一秒钟，所有相邻且互相瞄准的牛仔对（即A-B模式）会同时转身变成B-A。

当前观察到的状态是：{s}

请计算存在多少种不同的初始状态，经过一次时间演进后可以得到当前状态。注意：
1. 两个状态不同当且仅当至少有一个牛仔的指向不同
2. 结果请用<answer>数值</answer>标签包裹

例如，当状态是ABABAB时，正确答案是<answer>4</answer>"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'<answer>(\d+)</answer>', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
