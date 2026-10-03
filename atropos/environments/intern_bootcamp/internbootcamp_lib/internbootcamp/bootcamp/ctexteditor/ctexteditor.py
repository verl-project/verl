"""# 

### 谜题描述
Vasya is pressing the keys on the keyboard reluctantly, squeezing out his ideas on the classical epos depicted in Homer's Odysseus... How can he explain to his literature teacher that he isn't going to become a writer? In fact, he is going to become a programmer. So, he would take great pleasure in writing a program, but none — in writing a composition.

As Vasya was fishing for a sentence in the dark pond of his imagination, he suddenly wondered: what is the least number of times he should push a key to shift the cursor from one position to another one?

Let's describe his question more formally: to type a text, Vasya is using the text editor. He has already written n lines, the i-th line contains ai characters (including spaces). If some line contains k characters, then this line overall contains (k + 1) positions where the cursor can stand: before some character or after all characters (at the end of the line). Thus, the cursor's position is determined by a pair of integers (r, c), where r is the number of the line and c is the cursor's position in the line (the positions are indexed starting from one from the beginning of the line).

Vasya doesn't use the mouse to move the cursor. He uses keys \"Up\", \"Down\", \"Right\" and \"Left\". When he pushes each of these keys, the cursor shifts in the needed direction. Let's assume that before the corresponding key is pressed, the cursor was located in the position (r, c), then Vasya pushed key:

  * \"Up\": if the cursor was located in the first line (r = 1), then it does not move. Otherwise, it moves to the previous line (with number r - 1), to the same position. At that, if the previous line was short, that is, the cursor couldn't occupy position c there, the cursor moves to the last position of the line with number r - 1;
  * \"Down\": if the cursor was located in the last line (r = n), then it does not move. Otherwise, it moves to the next line (with number r + 1), to the same position. At that, if the next line was short, that is, the cursor couldn't occupy position c there, the cursor moves to the last position of the line with number r + 1;
  * \"Right\": if the cursor can move to the right in this line (c < ar + 1), then it moves to the right (to position c + 1). Otherwise, it is located at the end of the line and doesn't move anywhere when Vasya presses the \"Right\" key;
  * \"Left\": if the cursor can move to the left in this line (c > 1), then it moves to the left (to position c - 1). Otherwise, it is located at the beginning of the line and doesn't move anywhere when Vasya presses the \"Left\" key.



You've got the number of lines in the text file and the number of characters, written in each line of this file. Find the least number of times Vasya should push the keys, described above, to shift the cursor from position (r1, c1) to position (r2, c2).

Input

The first line of the input contains an integer n (1 ≤ n ≤ 100) — the number of lines in the file. The second line contains n integers a1, a2, ..., an (0 ≤ ai ≤ 105), separated by single spaces. The third line contains four integers r1, c1, r2, c2 (1 ≤ r1, r2 ≤ n, 1 ≤ c1 ≤ ar1 + 1, 1 ≤ c2 ≤ ar2 + 1).

Output

Print a single integer — the minimum number of times Vasya should push a key to move the cursor from position (r1, c1) to position (r2, c2).

Examples

Input

4
2 1 6 4
3 4 4 2


Output

3


Input

4
10 5 6 4
1 11 4 2


Output

6


Input

3
10 1 10
1 10 1 1


Output

3

Note

In the first sample the editor contains four lines. Let's represent the cursor's possible positions in the line as numbers. Letter s represents the cursor's initial position, letter t represents the last one. Then all possible positions of the cursor in the text editor are described by the following table.

123

12

123s567

1t345

One of the possible answers in the given sample is: \"Left\", \"Down\", \"Left\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long infl = 1e18 + 5;
long long int m, n, p, q, x, y, cn, k, tmp1, tmp,
    mx = -1, f, val, sz, sm, cnt, ans = infl, t = 1, i, j, ind = -1, ci;
long long int a[104], mn[104][104];
long long int getmin(int start, int end) {
  int i;
  long long int y = infl;
  for (i = start; i < end + 1; i++) {
    y = min(y, a[i]);
  }
  return y;
}
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  if (fopen(\"input.txt\", \"r\")) {
    freopen(\"output.txt\", \"w\", stdout);
    freopen(\"input.txt\", \"r\", stdin);
  }
  long long int r1, c1, r2, c2;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a[i];
    a[i]++;
  }
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      mn[i][j] = infl;
    }
  }
  cin >> r1 >> c1 >> r2 >> c2;
  r1--;
  r2--;
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (j == 0)
        mn[i][j] = a[i];
      else
        mn[i][j] = min(mn[i][j - 1], a[j]);
      mn[j][i] = mn[i][j];
    }
  }
  for (i = 0; i < n; i++) {
    if (i >= r1)
      sm = min(c1, getmin(r1 + 1, i));
    else
      sm = min(c1, getmin(i, r1 - 1));
    if (i <= r2)
      val = abs(r1 - i) + abs(r2 - i) + abs(c2 - min(sm, getmin(i + 1, r2)));
    else
      val = abs(r1 - i) + abs(r2 - i) + abs(c2 - min(sm, getmin(r2, i - 1)));
    ans = min(ans, val);
  }
  cout << ans;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp


class Ctexteditorbootcamp(Basebootcamp):
    def __init__(self, min_lines=2, max_lines=100, max_chars=10**5):
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.max_chars = max_chars

    def case_generator(self):
        n = random.randint(self.min_lines, self.max_lines)
        
        a = []
        for _ in range(n):
            if random.random() < 0.3:
                a.append(0)
            else:
                a.append(random.randint(0, self.max_chars))
        
        if random.random() < 0.5:
            r1 = r2 = random.randint(1, n)
        else:
            r1, r2 = random.sample(range(1, n+1), 2)
        
        max_c1 = a[r1-1] + 1
        c1 = random.randint(1, max(max_c1, 1))
        max_c2 = a[r2-1] + 1
        c2 = random.randint(1, max(max_c2, 1))
        
        return {
            'n': n,
            'a': a,
            'r1': r1,
            'c1': c1,
            'r2': r2,
            'c2': c2
        }

    @staticmethod
    def prompt_func(question_case):
        a_desc = []
        for idx, val in enumerate(question_case['a']):
            a_desc.append(f"第{idx+1}行：{val}字符（共{val+1}个光标位）")
        
        line_desc = '\n'.join(a_desc)
        
        prompt = f"""## 光标移动最小按键次数问题
文本编辑器共有{question_case['n']}行，各行的字符数如下：
{line_desc}

### 起始位置
- 行号：{question_case['r1']}
- 列号：{question_case['c1']}

### 目标位置
- 行号：{question_case['r2']}
- 列号：{question_case['c2']}

### 移动规则
1. 上下移动保持列号，若目标行不足该列则移动到行尾
2. 左右移动单步调整列号，无法越界移动

请计算最少按键次数，并将最终答案用[answer]标签包裹。"""

        return prompt

    @staticmethod
    def extract_output(output):
        import re
        patterns = [
            r'\[answer\](.*?)\[\/answer\]',
            r'最少需要[：: ]*(\d+)次按键',
            r'final answer:? (\d+)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output, re.DOTALL|re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        def min_steps_calculation(params):
            n = params['n']
            a = [x+1 for x in params['a']]
            r1, c1 = params['r1']-1, params['c1']
            r2, c2 = params['r2']-1, params['c2']
            
            min_table = [[float('inf')]*n for _ in range(n)]
            for i in range(n):
                current_min = a[i]
                min_table[i][i] = current_min
                for j in range(i+1, n):
                    current_min = min(current_min, a[j])
                    min_table[i][j] = current_min
                    min_table[j][i] = current_min
            
            min_operations = float('inf')
            for mid in range(n):
                up_min = min_table[mid][r1] if mid <= r1 else min_table[r1][mid]
                col_limit = min(c1, up_min)
                
                down_min = min_table[mid][r2] if mid <= r2 else min_table[r2][mid]
                final_col = min(col_limit, down_min)
                
                vertical = abs(mid - r1) + abs(mid - r2)
                horizontal = abs(final_col - c2)
                total = vertical + horizontal
                
                if total < min_operations:
                    min_operations = total
            
            return min_operations

        try:
            expected = min_steps_calculation(identity)
            return int(solution) == expected
        except (ValueError, KeyError, TypeError):
            return False
