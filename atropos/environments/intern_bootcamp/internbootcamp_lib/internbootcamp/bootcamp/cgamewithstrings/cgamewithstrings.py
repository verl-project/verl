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
using namespace std;
const int inf = 1e9 + 333;
const long long linf = 1e18 + 333;
const int N = 50;
const int M = 21;
int n, m, cnt[1 << M];
long long w[1 << M];
double dp[1 << M];
char s[N][M];
int main() {
  scanf(\"%d\", &n);
  for (int i = 0; i < n; i++) scanf(\"%s\", s[i]);
  m = strlen(s[0]);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      long long mask = 0;
      for (int k = 0; k < m; k++) mask |= (s[i][k] == s[j][k]) << k;
      w[mask] |= 1LL << i | 1LL << j;
    }
  }
  for (int i = (1 << m) - 1; i >= 0; i--)
    for (int j = 0; j < m; j++)
      if (i & 1 << j) w[i ^ 1 << j] |= w[i];
  for (int i = 0; i < (1 << m); i++) cnt[i] = __builtin_popcountll(w[i]);
  for (int i = (1 << m) - 1; i >= 0; i--) {
    if (!cnt[i]) continue;
    double sum = 0;
    for (int j = 0; j < m; j++)
      if (~i & 1 << j) sum += dp[i | 1 << j] * cnt[i | 1 << j];
    dp[i] = 1 + sum / (m - __builtin_popcount(i)) / cnt[i];
  }
  printf(\"%.18lf\n\", dp[0]);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from collections import defaultdict

class Cgamewithstringsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=50, min_m=1, max_m=20):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
    
    def case_generator(self):
        m = random.randint(self.min_m, self.max_m)
        n = random.randint(self.min_n, self.max_n)
        chars = string.ascii_letters
        strings = set()
        while len(strings) < n:
            s = ''.join(random.choice(chars) for _ in range(m))
            if s not in strings:
                strings.add(s)
        strings = list(strings)
        return {
            'n': n,
            'm': m,
            'strings': strings
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        strings = question_case['strings']
        input_lines = [str(n)] + strings
        input_str = '\n'.join(input_lines)
        prompt = f"""You are playing a game with your friend. The rules of the game are as follows:

Your friend creates {n} distinct strings of the same length and tells you all of them. He then randomly selects one string uniformly at random. Your task is to determine which string he selected by asking questions. Each question allows you to inquire about the character at a specific position in the string, which you haven't asked about before. You choose each position uniformly at random from the remaining positions. The process stops once the selected string is uniquely identified.

Your goal is to calculate the expected number of questions required to identify the chosen string.

Input format:
The first line contains an integer n (number of strings). The next n lines contain the distinct strings.

For example, the input may look like:
2
aab
aac

Your task is to compute the expected value, ensuring that the answer's absolute or relative error does not exceed 1e-9. Format your answer with at least 12 decimal places and enclose it within [answer] and [/answer] tags.

Input provided:
{input_str}

Please provide your answer within the specified tags."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return float(last_match)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        strings = identity['strings']
        try:
            correct_value = cls.calculate_expected_value(strings)
        except:
            return False
        if correct_value is None:
            return False
        if solution is None:
            return False
        absolute_error = abs(solution - correct_value)
        if absolute_error <= 1e-9:
            return True
        relative_error = absolute_error / (abs(correct_value) + 1e-12)
        return relative_error <= 1e-9
    
    @staticmethod
    def calculate_expected_value(strings):
        n = len(strings)
        if n == 0:
            return 0.0
        m = len(strings[0])
        if any(len(s) != m for s in strings):
            raise ValueError("All strings must have the same length")
        
        w = [0] * (1 << m)
        for i in range(n):
            for j in range(i):
                mask = 0
                for k in range(m):
                    if strings[i][k] == strings[j][k]:
                        mask |= 1 << k
                w[mask] |= (1 << i) | (1 << j)
        
        # Propagate the masks
        for mask in reversed(range(1 << m)):
            for k in range(m):
                if mask & (1 << k):
                    lower_mask = mask ^ (1 << k)
                    w[lower_mask] |= w[mask]
        
        cnt = [0] * (1 << m)
        for mask in range(1 << m):
            cnt[mask] = bin(w[mask]).count('1')
        
        dp = [0.0] * (1 << m)
        for mask in reversed(range(1 << m)):
            if cnt[mask] == 0:
                continue
            asked = bin(mask).count('1')
            remaining = m - asked
            if remaining == 0:
                dp[mask] = 0.0
                continue
            total = 0.0
            for k in range(m):
                if not (mask & (1 << k)):
                    next_mask = mask | (1 << k)
                    total += dp[next_mask] * cnt[next_mask]
            dp[mask] = 1 + total / (remaining * cnt[mask])
        return dp[0]
