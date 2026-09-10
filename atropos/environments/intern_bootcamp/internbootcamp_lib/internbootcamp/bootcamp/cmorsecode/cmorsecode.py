"""# 

### 谜题描述
In Morse code, an letter of English alphabet is represented as a string of some length from 1 to 4. Moreover, each Morse code representation of an English letter contains only dots and dashes. In this task, we will represent a dot with a \"0\" and a dash with a \"1\".

Because there are 2^1+2^2+2^3+2^4 = 30 strings with length 1 to 4 containing only \"0\" and/or \"1\", not all of them correspond to one of the 26 English letters. In particular, each string of \"0\" and/or \"1\" of length at most 4 translates into a distinct English letter, except the following four strings that do not correspond to any English alphabet: \"0011\", \"0101\", \"1110\", and \"1111\".

You will work with a string S, which is initially empty. For m times, either a dot or a dash will be appended to S, one at a time. Your task is to find and report, after each of these modifications to string S, the number of non-empty sequences of English letters that are represented with some substring of S in Morse code.

Since the answers can be incredibly tremendous, print them modulo 10^9 + 7.

Input

The first line contains an integer m (1 ≤ m ≤ 3 000) — the number of modifications to S. 

Each of the next m lines contains either a \"0\" (representing a dot) or a \"1\" (representing a dash), specifying which character should be appended to S.

Output

Print m lines, the i-th of which being the answer after the i-th modification to S.

Examples

Input

3
1
1
1


Output

1
3
7


Input

5
1
0
1
0
1


Output

1
4
10
22
43


Input

9
1
1
0
0
0
1
1
0
1


Output

1
3
10
24
51
109
213
421
833

Note

Let us consider the first sample after all characters have been appended to S, so S is \"111\".

As you can see, \"1\", \"11\", and \"111\" all correspond to some distinct English letter. In fact, they are translated into a 'T', an 'M', and an 'O', respectively. All non-empty sequences of English letters that are represented with some substring of S in Morse code, therefore, are as follows.

  1. \"T\" (translates into \"1\") 
  2. \"M\" (translates into \"11\") 
  3. \"O\" (translates into \"111\") 
  4. \"TT\" (translates into \"11\") 
  5. \"TM\" (translates into \"111\") 
  6. \"MT\" (translates into \"111\") 
  7. \"TTT\" (translates into \"111\") 



Although unnecessary for this task, a conversion table from English alphabets into Morse code can be found [here](https://en.wikipedia.org/wiki/Morse_code).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long MOD = 1e9 + 7;
const int INF = 0x3f3f3f3f;
const long long LINF = 0x3f3f3f3f3f3f3f3f;
int N, n, SA[3003], lcp[3003], arr[3003], rsa[3003];
long long D[3003][3003], T[3003];
char S[3003];
void SuffixArray() {
  int i, j, k;
  int m = 2;
  vector<int> cnt(max(N, m) + 1, 0), first(N + 1, 0), second(N + 1, 0);
  for (i = 1; i <= N; i++) cnt[first[i] = S[i] - 'a' + 1]++;
  for (i = 1; i <= m; i++) cnt[i] += cnt[i - 1];
  for (i = N; i; i--) SA[cnt[first[i]]--] = i;
  for (int len = 1, p = 1; p < N; len <<= 1, m = p) {
    for (p = 0, i = N - len; ++i <= N;) second[++p] = i;
    for (i = 1; i <= N; i++)
      if (SA[i] > len) second[++p] = SA[i] - len;
    for (i = 0; i <= m; i++) cnt[i] = 0;
    for (i = 1; i <= N; i++) cnt[first[second[i]]]++;
    for (i = 1; i <= m; i++) cnt[i] += cnt[i - 1];
    for (i = N; i; i--) SA[cnt[first[second[i]]]--] = second[i];
    swap(first, second);
    p = 1;
    first[SA[1]] = 1;
    for (i = 1; i < N; i++)
      first[SA[i + 1]] = SA[i] + len <= N && SA[i + 1] + len <= N &&
                                 second[SA[i]] == second[SA[i + 1]] &&
                                 second[SA[i] + len] == second[SA[i + 1] + len]
                             ? p
                             : ++p;
  }
}
void LCP() {
  int i, j, k = 0;
  vector<int> rank(N + 1, 0);
  for (i = 1; i <= N; i++) rank[SA[i]] = i;
  for (i = 1; i <= N; lcp[rank[i++]] = k)
    for (k ? k-- : 0, j = SA[rank[i] - 1]; S[i + k] == S[j + k]; k++)
      ;
}
char bstr[4][5] = {\"0011\", \"0101\", \"1110\", \"1111\"};
bool bad(int s, int e) {
  int i, j;
  for (j = 0; j < 4; j++) {
    for (i = s; i <= e; i++)
      if (bstr[j][i - s] - '0' != arr[i]) break;
    if (i == e + 1) return 1;
  }
  return 0;
}
int rns[3003];
int fin(int idx) {
  int i, p;
  for (i = 1; i <= n; i++)
    if (SA[i] == idx) break;
  p = i;
  int t = lcp[p];
  for (i = p - 1; i; i--) {
    rns[SA[i]] = t;
    t = min(t, lcp[i]);
  }
  t = lcp[p + 1];
  for (i = p + 1; i <= n; i++) {
    rns[SA[i]] = t;
    t = min(t, lcp[i + 1]);
  }
  int maxi = 0;
  for (i = 1; i < idx; i++) {
    maxi = max(maxi, rns[i]);
  }
  return idx - maxi;
}
int main() {
  int i, j, k;
  scanf(\"%d\", &n);
  N = n;
  for (i = 1; i <= n; i++) scanf(\"%d\", &arr[i]);
  for (i = n; i > 0; i--) S[n + 1 - i] = 'a' + arr[i];
  SuffixArray();
  LCP();
  for (i = 1; i <= n; i++) SA[i] = n + 1 - SA[i];
  for (i = 1; i <= n; i++) {
    for (j = 1; j <= i; j++) {
      for (k = 0; k <= 3 && k < i; k++) {
        if (k == 3 && bad(i - k, i)) continue;
        D[i][j] = (D[i][j] + D[i - k - 1][min(j, i - k - 1)]) % MOD;
        if (i - k <= j) D[i][j]++;
      }
    }
    int t = fin(i);
    T[i] = (T[i - 1] + D[i][t]) % MOD;
    printf(\"%lld\n\", T[i]);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7
INVALID_4 = {"0011", "0101", "1110", "1111"}

class Cmorsecodebootcamp(Basebootcamp):
    def __init__(self, m_max=3000):
        super().__init__()  # 添加父类初始化
        self.m_max = m_max

    def case_generator(self):
        m = random.randint(1, self.m_max)
        operations = [random.choice(['0', '1']) for _ in range(m)]
        S = ''.join(operations)
        
        # 动态规划计算正确结果
        dp = [0] * (m + 1)
        dp[0] = 1  # 空序列的初始化值
        sum_so_far = 0
        expected_outputs = []
        
        for k in range(1, m + 1):
            current = 0
            for l in range(1, 5):
                if k >= l:
                    start = k - l
                    substring = S[start:k]
                    # 检查4字符的无效情况
                    if l == 4 and substring in INVALID_4:
                        continue
                    current += dp[start]
                    current %= MOD
            dp[k] = current % MOD
            sum_so_far = (sum_so_far + dp[k]) % MOD
            expected_outputs.append(sum_so_far)
        
        return {
            'm': m,
            'operations': operations,
            'expected_outputs': expected_outputs
        }

    @staticmethod
    def prompt_func(question_case):
        m = question_case['m']
        ops = question_case['operations']
        input_str = f"{m}\n" + '\n'.join(ops)
        return f"""Given a sequence of Morse code operations, calculate the valid letter sequences after each step. Each letter is represented by a Morse string (0: dot, 1: dash) of length 1-4, excluding the forbidden patterns: 0011, 0101, 1110, 1111.

Input format:
{input_str}

Output {m} lines with the count modulo 1e9+7 after each step. Enclose your final answers within [answer] tags:

Example:
[answer]
42
[/answer]

[answer]
""" + "\n".join(map(str, question_case['expected_outputs'])) + "\n[/answer]"

    @staticmethod
    def extract_output(output):
        # 更鲁棒的数字提取逻辑
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        numbers = []
        for block in answer_blocks:
            numbers.extend(re.findall(r'\d+', block))
        
        try:
            return [int(num) % MOD for num in numbers]
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_outputs']
        m = identity['m']
        
        if not solution or len(solution) < m:
            return False
        
        # 取最后m个元素进行比较
        actual = solution[-m:]
        return all(
            (a % MOD) == (e % MOD) 
            for a, e in zip(actual, expected)
        )
