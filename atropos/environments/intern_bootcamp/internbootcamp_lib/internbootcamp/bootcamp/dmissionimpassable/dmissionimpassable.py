"""# 

### 谜题描述
Market stalls now have the long-awaited game The Colder Scrools V: Nvodsk. The game turned out to be difficult as hell and most students can't complete the last quest (\"We don't go to Nvodsk...\"). That threatened winter exams. The rector already started to wonder whether he should postpone the winter exams till April (in fact, he wanted to complete the quest himself). But all of a sudden a stranger appeared at the door of his office. \"Good afternoon. My name is Chuck and I solve any problems\" — he said.

And here they are sitting side by side but still they can't complete the mission. The thing is, to kill the final boss one should prove one's perfect skills in the art of managing letters. One should be a real magician to do that. And can you imagine what happens when magicians start competing... 

But let's put it more formally: you are given a string and a set of integers ai. You are allowed to choose any substring that is a palindrome and delete it. At that we receive some number of points equal to ak, where k is the length of the deleted palindrome. For some k, ak = -1, which means that deleting palindrome strings of such length is forbidden. After a substring is deleted, the remaining part \"shifts together\", that is, at no moment of time the string has gaps. The process is repeated while the string has at least one palindrome substring that can be deleted. All gained points are summed up.

Determine what maximum number of points can be earned.

\"Oh\" — said Chuck, raising from the chair, — \"I used to love deleting palindromes, just like you, but one day I took an arrow in the Knee\".

Input

The first line contains an integer l (1 ≤ l ≤ 150) — the length of the string.

The second line contains exactly l integers ak ( - 1 ≤ ak ≤ 105) — the points a player gains for deleting.

The third line contains exactly l lowercase Latin letters — the original string from which a player can delete palindromes. The line contains no other characters apart from the newline character at the end of the string.

Output

Print a single number — the maximum number of points one can gain if he plays on the given string.

Examples

Input

7
-1 -1 -1 -1 -1 -1 -1
abacaba


Output

0


Input

7
1 -1 -1 -1 -1 -1 -1
abacaba


Output

7


Input

7
1 5 -1 -1 -1 -1 10
abacaba


Output

16

Note

In the first sample we cannot delete any substring, so the best result is 0. In the second sample we are allowed to delete only those palindromes whose length equals 1, thus, if we delete the whole string, we get 7 points. In the third sample the optimal strategy is: first we delete character c, then string aa, then bb, and the last one aa. At that we get 1 + 3 * 5 = 16 points.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
char s[155];
const int INF = 1e9;
int a[155], best[155][155], f[155][155][155], flag, l, r, L, i, j, k, p, m, n;
void upd(int x) {
  if (x > f[i][j][k]) f[i][j][k] = x;
}
int main() {
  scanf(\"%d\", &n);
  for (i = 1; i <= n; i++) {
    scanf(\"%d\", &a[i]);
    if (a[i] == -1) a[i] = -INF;
  }
  scanf(\"%s\", s + 1);
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      for (k = 0; k <= n; k++) f[i][j][k] = -INF;
  for (L = 1; L <= n; L++)
    for (i = 1; i <= n - L + 1; i++) {
      j = i + L - 1;
      if (L == 1) f[i][j][0] = a[1];
      for (m = i; m < j; m++)
        f[i][j][0] = max(f[i][j][0], f[i][m][0] + f[m + 1][j][0]);
      for (flag = 1, l = i, r = j; l < r; l++, r--)
        if (s[l] != s[r]) flag = 0;
      if (flag) f[i][j][0] = max(f[i][j][0], a[L]);
      for (m = i; m <= j; m++)
        f[i][j][1] = max(f[i][j][1], f[i][m - 1][0] + f[m + 1][j][0]);
      for (k = 2; k <= L; k++) {
        for (p = i + 1; p <= j; p++) upd(f[i][p - 1][0] + f[p][j][k]);
        for (p = i; p < j; p++) upd(f[p + 1][j][0] + f[i][p][k]);
        if (s[i] == s[j]) upd(f[i + 1][j - 1][k - 2]);
      }
      for (k = 1; k <= L; k++) f[i][j][0] = max(f[i][j][0], f[i][j][k] + a[k]);
      best[i][j] = max(0, f[i][j][0]);
      for (m = i; m < j; m++)
        best[i][j] = max(best[i][j], best[i][m] + best[m + 1][j]);
    }
  printf(\"%d\n\", best[1][n]);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dmissionimpassablebootcamp(Basebootcamp):
    def __init__(self, max_length=50, a_positive_prob=0.7):
        self.max_length = max_length  # 降低默认最大长度
        self.a_positive_prob = a_positive_prob
    
    def case_generator(self):
        l = random.randint(3, self.max_length)
        # 生成包含回文的随机字符串
        core = random.choice(['aa', 'abba', 'abcba', 'aaa'])
        s = core + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=l-len(core)))
        s = ''.join(random.sample(s, len(s)))  # 打乱顺序保持回文可能性
        
        a = []
        for _ in range(l):
            if random.random() < self.a_positive_prob:
                a.append(random.randint(1, 100))
            else:
                a.append(-1)
        return {
            'length': l,
            'a': a,
            's': s[:l]
        }
    
    @staticmethod
    def prompt_func(question_case):
        l = question_case['length']
        a = question_case['a']
        s = question_case['s']
        a_str = ' '.join(map(str, a))
        return f"""根据游戏规则计算字符串{s}的最大可得分值。得分规则：
1. 删除长度为k的回文子串获得a[k-1]分（-1表示禁止删除）
2. 字符串自动拼接后持续删除直到无法操作
3. 输出最大值

输入：
长度：{l}
得分数组：{a_str}
字符串：{s}

将最终答案放在[answer]标签内，如[answer]16[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](-?\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _compute_max_score(cls, identity):
        s = identity['s']
        a = [x if x != -1 else -float('inf') for x in identity['a']]
        n = len(s)
        
        # Precompute palindrome table
        is_palin = [[False]*n for _ in range(n)]
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if i == j:
                    is_palin[i][j] = True
                elif i+1 == j:
                    is_palin[i][j] = (s[i] == s[j])
                else:
                    is_palin[i][j] = (s[i] == s[j] and is_palin[i+1][j-1])
        
        # Initialize DP tables
        dp = [[-float('inf')]*n for _ in range(n)]
        best = [[0]*n for _ in range(n)]
        
        for length in range(1, n+1):
            for i in range(n - length +1):
                j = i + length -1
                if length == 1:
                    dp[i][j] = a[0]
                else:
                    # Split into substrings
                    dp[i][j] = max([dp[i][k] + dp[k+1][j] for k in range(i, j)], default=-float('inf'))
                    
                    # Check entire palindrome
                    if is_palin[i][j]:
                        dp[i][j] = max(dp[i][j], a[length-1])
                
                # Update best solution
                best[i][j] = max(0, dp[i][j])
                for k in range(i, j):
                    best[i][j] = max(best[i][j], best[i][k] + best[k+1][j])
        
        return best[0][n-1]
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            correct = cls._compute_max_score(identity)
            return int(solution) == correct
        except:
            return False
