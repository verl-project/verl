"""# 

### 谜题描述
Sherlock Holmes found a mysterious correspondence of two VIPs and made up his mind to read it. But there is a problem! The correspondence turned out to be encrypted. The detective tried really hard to decipher the correspondence, but he couldn't understand anything. 

At last, after some thought, he thought of something. Let's say there is a word s, consisting of |s| lowercase Latin letters. Then for one operation you can choose a certain position p (1 ≤ p < |s|) and perform one of the following actions: 

  * either replace letter sp with the one that alphabetically follows it and replace letter sp + 1 with the one that alphabetically precedes it; 
  * or replace letter sp with the one that alphabetically precedes it and replace letter sp + 1 with the one that alphabetically follows it. 



Let us note that letter \"z\" doesn't have a defined following letter and letter \"a\" doesn't have a defined preceding letter. That's why the corresponding changes are not acceptable. If the operation requires performing at least one unacceptable change, then such operation cannot be performed.

Two words coincide in their meaning iff one of them can be transformed into the other one as a result of zero or more operations.

Sherlock Holmes needs to learn to quickly determine the following for each word: how many words can exist that coincide in their meaning with the given word, but differs from the given word in at least one character? Count this number for him modulo 1000000007 (109 + 7).

Input

The input data contains several tests. The first line contains the only integer t (1 ≤ t ≤ 104) — the number of tests.

Next t lines contain the words, one per line. Each word consists of lowercase Latin letters and has length from 1 to 100, inclusive. Lengths of words can differ.

Output

For each word you should print the number of different other words that coincide with it in their meaning — not from the words listed in the input data, but from all possible words. As the sought number can be very large, print its value modulo 1000000007 (109 + 7).

Examples

Input

1
ab


Output

1


Input

1
aaaaaaaaaaa


Output

0


Input

2
ya
klmbfxzb


Output

24
320092793

Note

Some explanations about the operation:

  * Note that for each letter, we can clearly define the letter that follows it. Letter \"b\" alphabetically follows letter \"a\", letter \"c\" follows letter \"b\", ..., \"z\" follows letter \"y\". 
  * Preceding letters are defined in the similar manner: letter \"y\" precedes letter \"z\", ..., \"a\" precedes letter \"b\". 
  * Note that the operation never changes a word's length. 



In the first sample you can obtain the only other word \"ba\". In the second sample you cannot obtain any other word, so the correct answer is 0.

Consider the third sample. One operation can transform word \"klmbfxzb\" into word \"klmcexzb\": we should choose p = 4, and replace the fourth letter with the following one (\"b\"  →  \"c\"), and the fifth one — with the preceding one (\"f\"  →  \"e\"). Also, we can obtain many other words from this one. An operation can transform word \"ya\" only into one other word \"xb\". 

Word \"ya\" coincides in its meaning with words \"xb\", \"wc\", \"vd\", ..., \"ay\" (overall there are 24 other words). The word \"klmbfxzb has many more variants — there are 3320092814 other words that coincide with in the meaning. So the answer for the first word equals 24 and for the second one equals 320092793 — the number 3320092814 modulo 109 + 7

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int mod = 1000000007;
int n;
int dp[102][2602];
char s[102];
int main() {
  dp[0][0] = 1;
  for (int i = 1; i <= 100; i++)
    for (int sum = 1; sum <= 2600; sum++)
      for (int j = 1; j <= 26 && j <= sum; j++) {
        dp[i][sum] += dp[i - 1][sum - j];
        if (dp[i][sum] >= mod) dp[i][sum] -= mod;
      }
  int test, res;
  scanf(\"%d\", &test);
  while (test--) {
    scanf(\"%s\", s);
    n = strlen(s);
    int sum = 0;
    for (int i = 0; i < n; i++) sum += s[i] - 'a' + 1;
    res = dp[n][sum] - 1;
    if (res < 0) res += mod;
    printf(\"%d\n\", res);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

# 动态规划预处理（优化循环顺序）
max_length = 100
max_sum = max_length * 26
dp = [[0] * (max_sum + 1) for _ in range(max_length + 1)]
dp[0][0] = 1
for i in range(1, max_length + 1):
    for s in range(1, max_sum + 1):
        for j in range(1, 27):
            if s >= j:
                dp[i][s] = (dp[i][s] + dp[i-1][s-j]) % MOD

class Ecipherbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_len = params.get('min_len', 1)
        self.max_len = params.get('max_len', 100)
        super().__init__(**params)

    def case_generator(self):
        """生成包含特殊边界案例的测试数据"""
        # 10%概率生成特殊案例
        if random.random() < 0.1:
            n = random.randint(self.min_len, self.max_len)
            # 生成全a或全z的极端情况
            word = random.choice([
                'a' * n,
                'z' * n,
                'a' + 'z'*(n-1) if n > 1 else 'a'
            ])
        else:
            n = random.randint(self.min_len, self.max_len)
            word = ''.join(random.choices(string.ascii_lowercase, k=n))
        return {"word": word}

    @staticmethod
    def prompt_func(question_case):
        word = question_case['word']
        return f"""请解决以下加密问题：
给定单词 "{word}"，计算可以通过合法操作转换得到的不同单词数量（需与原单词不同）。答案需为整数，并置于[answer]标签内。

操作规则：
1. 每次操作选择相邻两个字符(p和p+1)
2. 两种可选操作：
   a) p字符后移 + p+1字符前移
   b) p字符前移 + p+1字符后移
3. 非法操作示例：a不能前移，z不能后移

答案格式示例：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        """强化提取逻辑，处理多种数字格式"""
        matches = re.findall(r'\[answer\][\s]*(-?\d+)[\s]*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """修正验证逻辑"""
        try:
            word = identity['word']
            n = len(word)
            # 合法性检查
            if not (1 <= n <= 100) or not word.islower():
                return False
            
            # 计算字母值总和
            sum_val = sum(ord(c) - ord('a') + 1 for c in word)
            
            # 边界条件保证
            if not (n <= sum_val <= 26*n):
                return solution == 0  # 输入非法时应返回0
            
            # 数据库查询验证
            return (dp[n][sum_val] - 1) % MOD == solution
        except:
            return False
