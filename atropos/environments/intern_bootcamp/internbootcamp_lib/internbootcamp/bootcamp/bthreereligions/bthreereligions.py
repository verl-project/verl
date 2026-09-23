"""# 

### 谜题描述
During the archaeological research in the Middle East you found the traces of three ancient religions: First religion, Second religion and Third religion. You compiled the information on the evolution of each of these beliefs, and you now wonder if the followers of each religion could coexist in peace.

The Word of Universe is a long word containing the lowercase English characters only. At each moment of time, each of the religion beliefs could be described by a word consisting of lowercase English characters.

The three religions can coexist in peace if their descriptions form disjoint subsequences of the Word of Universe. More formally, one can paint some of the characters of the Word of Universe in three colors: 1, 2, 3, so that each character is painted in at most one color, and the description of the i-th religion can be constructed from the Word of Universe by removing all characters that aren't painted in color i.

The religions however evolve. In the beginning, each religion description is empty. Every once in a while, either a character is appended to the end of the description of a single religion, or the last character is dropped from the description. After each change, determine if the religions could coexist in peace.

Input

The first line of the input contains two integers n, q (1 ≤ n ≤ 100 000, 1 ≤ q ≤ 1000) — the length of the Word of Universe and the number of religion evolutions, respectively. The following line contains the Word of Universe — a string of length n consisting of lowercase English characters.

Each of the following line describes a single evolution and is in one of the following formats: 

  * + i c (i ∈ \{1, 2, 3\}, c ∈ \{a, b, ..., z\}: append the character c to the end of i-th religion description. 
  * - i (i ∈ \{1, 2, 3\}) – remove the last character from the i-th religion description. You can assume that the pattern is non-empty. 



You can assume that no religion will have description longer than 250 characters.

Output

Write q lines. The i-th of them should be YES if the religions could coexist in peace after the i-th evolution, or NO otherwise.

You can print each character in any case (either upper or lower).

Examples

Input


6 8
abdabc
+ 1 a
+ 1 d
+ 2 b
+ 2 c
+ 3 a
+ 3 b
+ 1 c
- 2


Output


YES
YES
YES
YES
YES
YES
NO
YES


Input


6 8
abbaab
+ 1 a
+ 2 a
+ 3 a
+ 1 b
+ 2 b
+ 3 b
- 1
+ 2 z


Output


YES
YES
YES
YES
YES
NO
YES
NO

Note

In the first example, after the 6th evolution the religion descriptions are: ad, bc, and ab. The following figure shows how these descriptions form three disjoint subsequences of the Word of Universe:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
void fast() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
}
vector<string> vec_splitter(string s) {
  s += ',';
  vector<string> res;
  while (!s.empty()) {
    res.push_back(s.substr(0, s.find(',')));
    s = s.substr(s.find(',') + 1);
  }
  return res;
}
void debug_out(vector<string> __attribute__((unused)) args,
               __attribute__((unused)) int idx,
               __attribute__((unused)) int LINE_NUM) {
  cerr << endl;
}
template <typename Head, typename... Tail>
void debug_out(vector<string> args, int idx, int LINE_NUM, Head H, Tail... T) {
  if (idx > 0)
    cerr << \", \";
  else
    cerr << \"Line(\" << LINE_NUM << \") \";
  stringstream ss;
  ss << H;
  cerr << args[idx] << \" = \" << ss.str();
  debug_out(args, idx + 1, LINE_NUM, T...);
}
double get_time() { return 1.0 * clock() / CLOCKS_PER_SEC; }
int main() {
  fast();
  int n, q;
  cin >> n >> q;
  string s;
  cin >> s;
  vector<vector<int>> nxt(26, vector<int>(n + 2, n + 1));
  for (int i = n - 1; i >= 0; i--) {
    nxt[s[i] - 'a'][i] = i;
  }
  for (int i = 0; i < 26; i++) {
    for (int j = n - 1; j >= 0; j--) nxt[i][j] = min(nxt[i][j], nxt[i][j + 1]);
  }
  vector<vector<vector<int>>> dp(
      256, vector<vector<int>>(256, vector<int>(256, n + 1)));
  dp[0][0][0] = 0;
  vector<int> l(3);
  vector<string> t(3, \"\");
  while (q--) {
    char ch, c;
    int idx;
    cin >> ch >> idx;
    idx--;
    if (ch == '+') {
      cin >> c;
      l[idx]++;
      t[idx] += c;
    }
    42;
    int lim0 = (idx == 0 ? l[0] : 0);
    int lim1 = (idx == 1 ? l[1] : 0);
    int lim2 = (idx == 2 ? l[2] : 0);
    for (int i = lim0; i <= l[0]; i++) {
      for (int j = lim1; j <= l[1]; j++) {
        for (int k = lim2; k <= l[2]; k++) {
          dp[i][j][k] = n + 1;
          if (ch == '+') {
            if (i > 0)
              dp[i][j][k] =
                  min(dp[i][j][k], nxt[t[0][i - 1] - 'a'][dp[i - 1][j][k]] + 1);
            if (j > 0)
              dp[i][j][k] =
                  min(dp[i][j][k], nxt[t[1][j - 1] - 'a'][dp[i][j - 1][k]] + 1);
            if (k > 0)
              dp[i][j][k] =
                  min(dp[i][j][k], nxt[t[2][k - 1] - 'a'][dp[i][j][k - 1]] + 1);
          }
        }
      }
    }
    if (ch == '-') {
      l[idx]--;
      t[idx] = t[idx].substr(0, (int)t[idx].size() - 1);
    }
    if (dp[l[0]][l[1]][l[2]] < n + 1)
      cout << \"YES\" << '\n';
    else
      cout << \"NO\" << '\n';
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import List, Dict, Any
from bootcamp import Basebootcamp

class Bthreereligionsbootcamp(Basebootcamp):
    def __init__(self, max_n: int = 100000, max_q: int = 1000, max_op_len: int = 250):
        self.max_n = max_n
        self.max_q = max_q
        self.max_op_len = max_op_len

    def case_generator(self) -> dict:
        n = random.randint(1, min(self.max_n, 100))
        q = random.randint(1, min(self.max_q, 50))
        word = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=n))
        operations = []
        current_lengths = [0, 0, 0]
        for _ in range(q):
            can_remove = any(l > 0 for l in current_lengths)
            if can_remove and random.random() < 0.5:
                valid_religions = [i+1 for i in range(3) if current_lengths[i] > 0]
                religion = random.choice(valid_religions)
                operations.append(f"- {religion}")
                current_lengths[religion-1] -= 1
            else:
                valid_religions = [i+1 for i in range(3) if current_lengths[i] < self.max_op_len]
                if not valid_religions:
                    valid_religions = [i+1 for i in range(3) if current_lengths[i] > 0]
                    if valid_religions:
                        religion = random.choice(valid_religions)
                        operations.append(f"- {religion}")
                        current_lengths[religion-1] -= 1
                    continue
                religion = random.choice(valid_religions)
                while current_lengths[religion-1] >= self.max_op_len:
                    valid_religions.remove(religion)
                    if not valid_religions:
                        break
                    religion = random.choice(valid_religions)
                if current_lengths[religion-1] >= self.max_op_len:
                    continue
                c = random.choice('abcdefghijklmnopqrstuvwxyz')
                operations.append(f"+ {religion} {c}")
                current_lengths[religion-1] += 1
        expected_outputs = self.get_expected_outputs(word, operations)
        return {
            "word": word,
            "operations": operations,
            "expected_outputs": expected_outputs
        }

    @staticmethod
    def get_expected_outputs(word: str, operations: List[str]) -> List[str]:
        n = len(word)
        nxt = [[n+1]*(n+2) for _ in range(26)]
        for i in range(n-1, -1, -1):
            c = ord(word[i]) - ord('a')
            nxt[c][i] = i
        for c in range(26):
            for j in range(n-1, -1, -1):
                if nxt[c][j] == n+1:
                    nxt[c][j] = nxt[c][j+1]
        dp = [[[n+1 for _ in range(251)] for __ in range(251)] for ___ in range(251)]
        dp[0][0][0] = 0
        l = [0, 0, 0]
        t = ['', '', '']
        expected_outputs = []
        for op in operations:
            parts = op.split()
            if parts[0] == '+':
                religion = int(parts[1]) - 1
                c = parts[2]
                t[religion] += c
                l[religion] += 1
                lim = [0, 0, 0]
                lim[religion] = l[religion]
                for i in range(lim[0], l[0]+1):
                    for j in range(lim[1], l[1]+1):
                        for k in range(lim[2], l[2]+1):
                            if i + j + k == 0:
                                continue
                            current_min = n+1
                            if i > 0:
                                pos = dp[i-1][j][k]
                                if pos <= n:
                                    char = t[0][i-1]
                                    new_pos = nxt[ord(char) - ord('a')][pos]
                                    if new_pos < n+1:
                                        current_min = min(current_min, new_pos + 1)
                            if j > 0:
                                pos = dp[i][j-1][k]
                                if pos <= n:
                                    char = t[1][j-1]
                                    new_pos = nxt[ord(char) - ord('a')][pos]
                                    if new_pos < n+1:
                                        current_min = min(current_min, new_pos + 1)
                            if k > 0:
                                pos = dp[i][j][k-1]
                                if pos <= n:
                                    char = t[2][k-1]
                                    new_pos = nxt[ord(char) - ord('a')][pos]
                                    if new_pos < n+1:
                                        current_min = min(current_min, new_pos + 1)
                            dp[i][j][k] = current_min
            else:
                religion = int(parts[1]) - 1
                t[religion] = t[religion][:-1]
                l[religion] -= 1
            current_dp = dp[l[0]][l[1]][l[2]]
            expected_outputs.append("YES" if current_dp <= n else "NO")
        return expected_outputs

    @staticmethod
    def prompt_func(question_case: dict) -> str:
        problem_desc = f"""你是一位考古学家，发现了三个古代宗教的痕迹。这三个宗教的演变可以通过其描述字符串的增减来观察。每次演变后，你需要判断这三个描述是否能作为互不相交的子序列存在于给定的宇宙之词中。

宇宙之词是一个仅包含小写字母的字符串。每个宗教的描述字符串初始为空，每次演变可以添加或删除一个字符。三个宗教的描述字符串能和平共存的条件是：它们的字符可以互不重叠地作为子序列存在于宇宙之词中。即，每个字符在宇宙之词中只能被一个宗教使用。

现在，给你一个宇宙之词和一系列的演变操作，每个操作后你需要判断是否满足条件。

输入格式：

第一行是宇宙之词的长度n和演变次数q。第二行是宇宙之词。随后的q行，每行描述一个演变操作，格式为“+ i c”或“- i”，其中i是宗教编号（1、2、3），c是小写字母。

输出格式：

对于每个操作后，输出YES如果三个宗教的描述可以和平共存，否则输出NO。

现在，请解决以下具体问题：

宇宙之词是：{question_case['word']}

演变操作：\n"""
        for op in question_case['operations']:
            problem_desc += f"{op}\n"
        problem_desc += """\n你需要输出每个操作后的结果。将你的答案放在[answer]和[/answer]标记之间，每个结果占一行，按顺序排列。"""
        return problem_desc

    @staticmethod
    def extract_output(output: str) -> list:
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL | re.IGNORECASE)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_block.split('\n') if line.strip()]
        results = []
        for line in lines:
            line_upper = line.upper()
            if line_upper in ('YES', 'NO'):
                results.append(line_upper)
            else:
                return None
        return results if results else None

    @classmethod
    def _verify_correction(cls, solution: list, identity: dict) -> bool:
        expected = identity['expected_outputs']
        if not solution or len(solution) != len(expected):
            return False
        return all(s == e.upper() for s, e in zip(solution, expected))
