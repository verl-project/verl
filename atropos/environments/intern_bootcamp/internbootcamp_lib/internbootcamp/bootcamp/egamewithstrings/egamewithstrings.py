"""# 

### 谜题描述
You play the game with your friend. The description of this game is listed below. 

Your friend creates n distinct strings of the same length m and tells you all the strings. Then he randomly chooses one of them. He chooses strings equiprobably, i.e. the probability of choosing each of the n strings equals <image>. You want to guess which string was chosen by your friend. 

In order to guess what string your friend has chosen, you are allowed to ask him questions. Each question has the following form: «What character stands on position pos in the string you have chosen?» A string is considered guessed when the answers to the given questions uniquely identify the string. After the string is guessed, you stop asking questions. 

You do not have a particular strategy, so as each question you equiprobably ask about a position that hasn't been yet mentioned. Your task is to determine the expected number of questions needed to guess the string chosen by your friend.

Input

The first line contains a single integer n (1 ≤ n ≤ 50) — the number of strings your friend came up with.

The next n lines contain the strings that your friend has created. It is guaranteed that all the strings are distinct and only consist of large and small English letters. Besides, the lengths of all strings are the same and are between 1 to 20 inclusive.

Output

Print the single number — the expected value. Your answer will be considered correct if its absolute or relative error doesn't exceed 10 - 9.

Examples

Input

2
aab
aac


Output

2.000000000000000


Input

3
aaA
aBa
Caa


Output

1.666666666666667


Input

3
aca
vac
wqq


Output

1.000000000000000

Note

In the first sample the strings only differ in the character in the third position. So only the following situations are possible: 

  * you guess the string in one question. The event's probability is <image>; 
  * you guess the string in two questions. The event's probability is <image> · <image> = <image> (as in this case the first question should ask about the position that is other than the third one); 
  * you guess the string in three questions. The event's probability is <image> · <image> · <image> = <image>; 



Thus, the expected value is equal to <image>

In the second sample we need at most two questions as any pair of questions uniquely identifies the string. So the expected number of questions is <image>.

In the third sample whatever position we ask about in the first question, we immediately identify the string.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
#pragma comment(linker, \"/STACK:100000000000000\")
using namespace std;
int n, vis[(1 << 21) + 1], col = 1, m, tw[22], cnt[(1 << 21) + 1];
long long dp[(1 << 21) + 1];
string st[55];
long long rep[(1 << 21) + 1];
int count_bit(long long x) {
  int ret = 0;
  while (x) {
    ret++;
    x ^= x & (-x);
  }
  return ret;
}
int main() {
  cin >> n;
  if (n == 1) {
    printf(\"%.10lf\n\", 0.);
    return 0;
  }
  for (int i = 0; i < 22; i++) tw[i] = 1 << i;
  for (int i = 0; i < n; i++) cin >> st[i];
  m = st[0].size();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i != j) {
        int msk = 0;
        for (int z = 0; z < m; z++)
          if (st[i][z] == st[j][z]) msk |= tw[z];
        rep[msk] |= 1ll << i;
      }
  for (int msk = tw[m] - 1; msk >= 0; msk--)
    for (int i = 0; i < m; i++)
      if (msk & tw[i]) {
        rep[msk ^ tw[i]] |= rep[msk];
      }
  for (int msk = tw[m + 1] - 1; msk >= 0; msk--) cnt[msk] = count_bit(msk);
  for (int msk = 0; msk < tw[m]; msk++) {
    int c = cnt[msk];
    for (int i = 0; i < m; i++)
      if (!(msk & tw[i])) {
        long long tmp = rep[msk] ^ rep[msk ^ tw[i]];
        dp[c + 1] += count_bit(tmp);
      }
  }
  double ans = 0;
  for (int i = 1; i < m + 1; i++) {
    double tmp = double(dp[i] * i) / (m - i + 1);
    for (int j = 0; j < i - 1; j++) tmp *= double(i - 1 - j) / (m - j);
    ans += tmp;
  }
  ans /= n;
  printf(\"%.10lf\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import string
from collections import defaultdict
from math import isclose
from bootcamp import Basebootcamp

class Egamewithstringsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=50, min_m=1, max_m=20):
        self.min_n = max(1, min(min_n, max_n))
        self.max_n = min(max_n, 50)
        self.min_m = max(1, min(min_m, max_m))
        self.max_m = min(max_m, 20)
        
        if self.min_n > self.max_n:
            self.min_n = self.max_n
        if self.min_m > self.max_m:
            self.min_m = self.max_m

    def case_generator(self):
        while True:
            m = random.randint(self.min_m, self.max_m)
            max_unique = (52) ** m  # 26大写+26小写字母的组合数
            safe_max_n = min(self.max_n, max_unique)
            
            if safe_max_n < self.min_n:
                continue  # 跳过无法生成的情况
                
            n = random.randint(self.min_n, safe_max_n)
            
            charset = string.ascii_letters
            strings = []
            existing = set()
            attempts = 0
            max_attempts = 1000
            
            while len(strings) < n and attempts < max_attempts:
                s = ''.join(random.choices(charset, k=m))
                if s not in existing:
                    existing.add(s)
                    strings.append(s)
                attempts += 1
                
            if len(strings) == n:
                return {'n': n, 'strings': strings}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        strings = question_case['strings']
        case_lines = '\n'.join(strings)
        m = len(strings[0])
        return f"""## String Guessing Game Probability Challenge

**Game Rules:**
1. Your friend creates {n} distinct {m}-character strings using uppercase and lowercase letters
2. Randomly selects one string with equal probability
3. You ask position-specific questions to identify the chosen string
4. Each question:
   - Randomly selects an unasked character position
   - Reveals the character at that position
5. Stop when answers uniquely identify the string

**Input Example:**
{n}
{case_lines}

**Your Task:**
Calculate the expected number of questions needed. Format requirements:
- 15 decimal places minimum
- Place final answer within [answer] tags like: [answer]1.234567890123456[/answer]
- Ensure absolute/relative error ≤ 1e-9

**Sample Calculation:**
For 2 strings "aab" and "aac":
- 1 question if ask position 3 (prob 1/3)
- 2 questions if first question misses position 3 (prob 2/3 * 1/2)
- Expected = 1*(1/3) + 2*(2/3*1/2) + 3*(1/3) = 2.0"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return float(matches[-1].strip().split()[0])
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = cls.calculate_expected(identity['n'], identity['strings'])
            user_value = float(solution)
            return isclose(user_value, expected, rel_tol=1e-9, abs_tol=1e-12)
        except:
            return False

    @staticmethod
    def calculate_expected(n, strings):
        if n == 1:
            return 0.0
        
        m = len(strings[0])
        tw = [1 << i for i in range(m)]
        rep = defaultdict(int)

        # 建立特征掩码映射
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                mask = 0
                for z in range(m):
                    if strings[i][z] == strings[j][z]:
                        mask |= tw[z]
                rep[mask] |= (1 << i)

        # 传播特征掩码
        for mask in reversed(range(1 << m)):
            for z in range(m):
                if (mask >> z) & 1:
                    lower_mask = mask ^ (1 << z)
                    rep[lower_mask] |= rep[mask]

        # 预计算位数
        cnt = [bin(bits).count('1') for bits in range(1 << m)]
        dp = [0] * (m + 2)

        # 动态规划处理状态转移
        for mask in range(1 << m):
            current_bits = cnt[mask]
            for z in range(m):
                if not (mask & (1 << z)):
                    new_mask = mask | (1 << z)
                    changed = rep[mask] ^ rep[new_mask]
                    dp[current_bits + 1] += bin(changed).count('1')

        # 计算最终期望值
        ans = 0.0
        for i in range(1, m + 1):
            if dp[i] == 0:
                continue
            
            probability = dp[i] / (n * (m - i + 1))
            for j in range(1, i):
                probability *= (i - j) / (m - j + 1)
            
            ans += probability * i

        return ans
