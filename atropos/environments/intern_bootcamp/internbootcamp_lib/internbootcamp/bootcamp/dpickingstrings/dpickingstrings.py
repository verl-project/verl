"""# 

### 谜题描述
Alice has a string consisting of characters 'A', 'B' and 'C'. Bob can use the following transitions on any substring of our string in any order any number of times: 

  * A <image> BC
  * B <image> AC
  * C <image> AB
  * AAA <image> empty string 



Note that a substring is one or more consecutive characters. For given queries, determine whether it is possible to obtain the target string from source.

Input

The first line contains a string S (1 ≤ |S| ≤ 105). The second line contains a string T (1 ≤ |T| ≤ 105), each of these strings consists only of uppercase English letters 'A', 'B' and 'C'.

The third line contains the number of queries Q (1 ≤ Q ≤ 105).

The following Q lines describe queries. The i-th of these lines contains four space separated integers ai, bi, ci, di. These represent the i-th query: is it possible to create T[ci..di] from S[ai..bi] by applying the above transitions finite amount of times?

Here, U[x..y] is a substring of U that begins at index x (indexed from 1) and ends at index y. In particular, U[1..|U|] is the whole string U.

It is guaranteed that 1 ≤ a ≤ b ≤ |S| and 1 ≤ c ≤ d ≤ |T|.

Output

Print a string of Q characters, where the i-th character is '1' if the answer to the i-th query is positive, and '0' otherwise.

Example

Input

AABCCBAAB
ABCB
5
1 3 1 2
2 2 2 4
7 9 1 1
3 4 2 3
4 5 1 3


Output

10011

Note

In the first query we can achieve the result, for instance, by using transitions <image>.

The third query asks for changing AAB to A — but in this case we are not able to get rid of the character 'B'.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
#pragma GCC target(\"pclmul\")
using ul = std::uint32_t;
using ull = std::uint64_t;
using li = std::int32_t;
using ll = std::int64_t;
using uss = std::uint8_t;
const ul maxn = 1e5;
char str[maxn + 2];
ul scb[maxn + 1];
ul sca[maxn + 1];
ul tcb[maxn + 1];
ul tca[maxn + 1];
ul q;
int main() {
  std::scanf(\"%s\", str + 1);
  for (ul i = 1; str[i]; ++i) {
    if (str[i] == 'A') {
      sca[i] = sca[i - 1] + 1;
      scb[i] = scb[i - 1];
    } else {
      scb[i] = scb[i - 1] + 1;
    }
  }
  std::scanf(\"%s\", str + 1);
  for (ul i = 1; str[i]; ++i) {
    if (str[i] == 'A') {
      tca[i] = tca[i - 1] + 1;
      tcb[i] = tcb[i - 1];
    } else {
      tcb[i] = tcb[i - 1] + 1;
    }
  }
  std::scanf(\"%u\", &q);
  for (ul i = 1; i <= q; ++i) {
    ul a, b, c, d;
    std::scanf(\"%u%u%u%u\", &a, &b, &c, &d);
    ul sb = scb[b] - scb[a - 1];
    ul tb = tcb[d] - tcb[c - 1];
    ul sa = std::min(sca[b], b - a + 1);
    ul ta = std::min(tca[d], d - c + 1);
    if ((sb ^ tb) & 1) {
      std::putchar('0');
      continue;
    }
    if (sa < ta) {
      std::putchar('0');
      continue;
    }
    if ((sa - ta) % 3 == 0 && sb > tb) {
      std::putchar('0');
      continue;
    }
    if ((sa - ta) % 3 != 0 && sb + 2 > tb) {
      std::putchar('0');
      continue;
    }
    if (sb == 0 && tb != 0 && sa == ta) {
      std::putchar('0');
      continue;
    }
    std::putchar('1');
  }
  std::putchar('\n');
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_prefixes(s):
    sca = [0] * (len(s) + 1)
    scb = [0] * (len(s) + 1)
    for i in range(1, len(s) + 1):
        char = s[i-1]
        if char == 'A':
            sca[i] = sca[i-1] + 1
            scb[i] = scb[i-1]
        else:
            sca[i] = sca[i-1]
            scb[i] = scb[i-1] + 1
    return sca, scb

def compute_query_result(sca, scb, tca, tcb, a, b, c, d):
    sb = scb[b] - scb[a-1]
    sa = min(sca[b], b - a + 1)
    tb = tcb[d] - tcb[c-1]
    ta = min(tca[d], d - c + 1)

    if (sb ^ tb) & 1:
        return '0'
    if sa < ta:
        return '0'
    if (sa - ta) % 3 == 0:
        if sb > tb:
            return '0'
    else:
        if (sb + 2) > tb:
            return '0'
    if sb == 0 and tb != 0 and sa == ta:
        return '0'
    return '1'

class Dpickingstringsbootcamp(Basebootcamp):
    def __init__(self, max_s_length=10, max_t_length=10, max_queries=5, max_attempts=10):
        self.max_s_length = max_s_length
        self.max_t_length = max_t_length
        self.max_queries = max_queries
        self.max_attempts = max_attempts
    
    def case_generator(self):
        for _ in range(self.max_attempts):
            s = ''.join(random.choices(['A', 'B', 'C'], k=random.randint(1, self.max_s_length)))
            t = ''.join(random.choices(['A', 'B', 'C'], k=random.randint(1, self.max_t_length)))
            sca, scb = compute_prefixes(s)
            tca, tcb = compute_prefixes(t)
            a, b = 1, len(s)
            c, d = 1, len(t)
            if compute_query_result(sca, scb, tca, tcb, a, b, c, d) == '1':
                queries = [[a, b, c, d]]
                for _ in range(self.max_queries - 1):
                    a_q = random.randint(1, len(s))
                    b_q = random.randint(a_q, len(s))
                    c_q = random.randint(1, len(t))
                    d_q = random.randint(c_q, len(t))
                    queries.append([a_q, b_q, c_q, d_q])
                return {
                    'S': s,
                    'T': t,
                    'queries': queries
                }
        # Fallback if no valid case found
        s = ''.join(random.choices(['A', 'B', 'C'], k=random.randint(1, self.max_s_length)))
        t = ''.join(random.choices(['A', 'B', 'C'], k=random.randint(1, self.max_t_length)))
        queries = []
        for _ in range(self.max_queries):
            a = random.randint(1, len(s))
            b = random.randint(a, len(s))
            c = random.randint(1, len(t))
            d = random.randint(c, len(t))
            queries.append([a, b, c, d])
        return {
            'S': s,
            'T': t,
            'queries': queries
        }
    
    @staticmethod
    def prompt_func(question_case):
        s = question_case['S']
        t = question_case['T']
        queries = question_case['queries']
        prompt = f"""Alice有一个字符串S，内容为：{s}。Bob可以对该字符串的任意子串应用以下转换规则（次数不限、顺序不限）：

- 将单个'A'替换为'BC'；
- 将单个'B'替换为'AC'；
- 将单个'C'替换为'AB'；
- 将连续的三个'A'（即"AAA"）替换为空字符串（即删除）。

现有目标字符串T，内容为：{t}。你需要处理{len(queries)}个查询，每个查询询问是否存在一种可能，通过应用上述规则，将S的某个子串转换为T的对应子串。

每个查询的参数为四个整数a、b、c、d，表示：源子串是S的第a到第b个字符（包含两端，按1-based索引），目标子串是T的第c到第d个字符。要求判断是否可以进行这样的转换。

请依次对每个查询给出“是”（用'1'表示）或“否”（用'0'表示）的答案，并将所有答案按顺序组成一个连续的字符串，无需分隔符。

请将最终答案放置在[answer]和[/answer]标签之间。例如：[answer]101[/answer]。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        s = identity['S']
        t = identity['T']
        queries = identity['queries']
        sca, scb = compute_prefixes(s)
        tca, tcb = compute_prefixes(t)
        expected = []
        for a, b, c, d in queries:
            res = compute_query_result(sca, scb, tca, tcb, a, b, c, d)
            expected.append(res)
        expected_str = ''.join(expected)
        return solution == expected_str
