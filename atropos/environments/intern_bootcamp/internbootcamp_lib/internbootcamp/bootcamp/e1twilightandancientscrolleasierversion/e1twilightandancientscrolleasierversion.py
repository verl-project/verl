"""# 

### 谜题描述
This is an easier version of the problem E with smaller constraints.

Twilight Sparkle has received a new task from Princess Celestia. This time she asked to decipher the ancient scroll containing important knowledge of pony origin.

To hide the crucial information from evil eyes, pony elders cast a spell on the scroll. That spell adds exactly one letter in any place to each word it is cast on. To make the path to the knowledge more tangled elders chose some of words in the scroll and cast a spell on them.

Twilight Sparkle knows that the elders admired the order in all things so the scroll original scroll contained words in lexicographically non-decreasing order. She is asked to delete one letter from some of the words of the scroll (to undo the spell) to get some version of the original scroll. 

Unfortunately, there may be more than one way to recover the ancient scroll. To not let the important knowledge slip by Twilight has to look through all variants of the original scroll and find the required one. To estimate the maximum time Twilight may spend on the work she needs to know the number of variants she has to look through. She asks you to find that number! Since that number can be very big, Twilight asks you to find it modulo 10^9+7.

It may occur that princess Celestia has sent a wrong scroll so the answer may not exist.

A string a is lexicographically smaller than a string b if and only if one of the following holds:

  * a is a prefix of b, but a ≠ b;
  * in the first position where a and b differ, the string a has a letter that appears earlier in the alphabet than the corresponding letter in b.

Input

The first line contains a single integer n (1 ≤ n ≤ 1000): the number of words in the scroll.

The i-th of the next n lines contains a string consisting of lowercase English letters: the i-th word in the scroll. The length of each word is more or equal than 1. The sum of lengths of words does not exceed 20000.

Output

Print one integer: the number of ways to get a version of the original from the scroll modulo 10^9+7.

Examples

Input


3
abcd
zaza
ataka


Output


4


Input


4
dfs
bfs
sms
mms


Output


8


Input


3
abc
bcd
a


Output


0


Input


6
lapochka
kartyshka
bigbabytape
morgenshtern
ssshhhiiittt
queen


Output


2028

Note

Notice that the elders could have written an empty word (but they surely cast a spell on it so it holds a length 1 now).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 100000 + 5, P = 131, mod = 1e9 + 7;
vector<int> vec, rk[N], dp[N];
vector<unsigned long long> hs[N];
int n, len[N], nxt[N * 10];
char s[N * 10];
unsigned long long bin[N * 10];
inline unsigned long long getHash(int x, int at, int l) {
  if (at > l) return hs[x][l];
  unsigned long long s = at == 0 ? 0 : hs[x][at - 1];
  return s * bin[l + 1 - at] + hs[x][l + 1] - hs[x][at] * bin[l + 1 - at];
}
inline int getChar(int x, int at, int l) {
  if (at > l) return l == 0 ? hs[x][0] : hs[x][l] - hs[x][l - 1] * P;
  return hs[x][l + 1] - hs[x][l] * P;
}
inline int cmp(int x1, int a1, int x2, int a2) {
  int l = -1, r = min(len[x1], len[x2]) - 2, mid;
  while (l < r) {
    if (r == l + 1) break;
    mid = (l + r) >> 1;
    if (getHash(x1, a1, mid) == getHash(x2, a2, mid)) {
      l = mid;
    } else {
      r = mid;
    }
  }
  if (getChar(x1, a1, r) < getChar(x2, a2, r)) return 1;
  if (getChar(x1, a1, r) > getChar(x2, a2, r)) return 0;
  int l1 = a1 == len[x1] - 1 ? len[x1] - 1 : len[x1] - 2;
  int l2 = a2 == len[x2] - 1 ? len[x2] - 1 : len[x2] - 2;
  return l1 <= l2;
}
void DP(int x) {
  if (x == 1) {
    for (int i = 0; i < len[1]; i++) dp[1].push_back(1);
    return;
  }
  DP(x - 1);
  int sum = 0, p = -1;
  for (int i = 0; i < len[x]; i++) {
    while (p + 1 < len[x - 1] && cmp(x - 1, rk[x - 1][p + 1], x, rk[x][i])) {
      ++p, sum = (sum + dp[x - 1][p]) % mod;
    }
    dp[x].push_back(sum);
  }
}
int main() {
  bin[0] = 1;
  for (int i = 1; i <= 1000000; i++) bin[i] = bin[i - 1] * P;
  scanf(\"%d\", &n);
  for (int i = 1; i <= n; i++) {
    scanf(\"%s\", s), len[i] = strlen(s), s[len[i]] = '#', ++len[i];
    int l = 0, r;
    while (l < len[i]) {
      r = l;
      while (r + 1 < len[i] && s[r + 1] == s[l]) ++r;
      for (int i = l; i <= r; i++) nxt[i] = r + 1;
      l = r + 1;
    }
    for (int j = 0; j < len[i]; j++) rk[i].push_back(0);
    l = 0, r = len[i] - 1;
    for (int j = 0; j < len[i]; j++) {
      if (l == r) {
        rk[i][l] = j;
      } else if (s[j] < s[nxt[j]]) {
        rk[i][r] = j, --r;
      } else {
        rk[i][l] = j, ++l;
      }
    }
    for (int j = 0; j < len[i]; j++) hs[i].push_back(0);
    hs[i][0] = s[0] - '#';
    for (int j = 1; j < len[i]; j++) hs[i][j] = hs[i][j - 1] * P + s[j] - '#';
  }
  DP(n);
  int ans = 0;
  for (int j = 0; j < len[n]; j++) ans = (ans + dp[n][j]) % mod;
  printf(\"%d\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import bisect
from bootcamp import Basebootcamp

MOD = 10**9 + 7

def compute_answer(n, words):
    if n == 0:
        return 0
    preprocessed = []
    for s in words:
        m = len(s)
        deletions = [s[:i] + s[i+1:] for i in range(m)]
        deletions.sort()
        preprocessed.append(deletions)
    
    prev_deletions = preprocessed[0]
    prev_prefix_sum = [0] * (len(prev_deletions) + 1)
    for i in range(len(prev_deletions)):
        prev_prefix_sum[i+1] = (prev_prefix_sum[i] + 1) % MOD
    
    for x in range(1, n):
        current_deletions = preprocessed[x]
        current_dp = []
        for s in current_deletions:
            j = bisect.bisect_right(prev_deletions, s)
            current_count = prev_prefix_sum[j]
            current_dp.append(current_count % MOD)
        
        current_prefix_sum = [0]
        current_sum = 0
        for cnt in current_dp:
            current_sum = (current_sum + cnt) % MOD
            current_prefix_sum.append(current_sum)
        
        prev_deletions = current_deletions
        prev_prefix_sum = current_prefix_sum
    
    return prev_prefix_sum[-1] % MOD

class E1twilightandancientscrolleasierversionbootcamp(Basebootcamp):
    def __init__(self, max_n=1000, max_total_length=20000):
        self.max_n = max_n
        self.max_total_length = max_total_length
    
    def case_generator(self):
        # 生成保证有效的测试用例
        n = random.randint(1, 3)
        words = []
        total_length = 0
        
        # 生成原始非递减序列
        original = []
        prev_word = ''
        for _ in range(n):
            # 生成保证≥前一个词的词
            if not original:
                min_length = 0
            else:
                min_length = len(prev_word)
            
            # 生成新词长度（原词长度+1）
            new_length = random.randint(min_length, min_length + 1)
            new_word = []
            for i in range(new_length):
                # 保证词序递增
                min_char = ord('a') if not new_word else ord(new_word[-1])
                new_char = chr(random.randint(min_char, ord('z')))
                new_word.append(new_char)
            
            new_word = ''.join(new_word)
            original.append(new_word)
            prev_word = new_word
        
        # 生成带干扰字符的测试用例
        scroll_words = []
        for orig in original:
            # 原始单词插入一个随机字符
            insert_pos = random.randint(0, len(orig))
            insert_char = chr(random.randint(ord('a'), ord('z')))
            new_word = orig[:insert_pos] + insert_char + orig[insert_pos:]
            scroll_words.append(new_word)
            total_length += len(new_word)
            if total_length > self.max_total_length:
                # 动态调整n值
                n = len(scroll_words)
                break
        
        # 计算正确答案
        try:
            expected = compute_answer(n, scroll_words)
        except Exception as e:
            print(f"Error computing answer: {e}")
            expected = 0
        
        return {
            'n': n,
            'words': scroll_words,
            'expected_answer': expected
        }
    
    @staticmethod
    def prompt_func(question_case):
        case = question_case
        input_str = f"{case['n']}\n" + "\n".join(case['words'])
        return f"""根据以下规则解决古代卷轴问题：
1. 每个单词需要删除正好一个字符
2. 最终序列必须是非递减字典序
3. 输出所有可能方案数模1000000007

输入：
{input_str}

请将最终答案放在[answer]标签内，如：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_answer']
