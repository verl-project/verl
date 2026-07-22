"""# 

### 谜题描述
Sereja has a bracket sequence s1, s2, ..., sn, or, in other words, a string s of length n, consisting of characters \"(\" and \")\".

Sereja needs to answer m queries, each of them is described by two integers li, ri (1 ≤ li ≤ ri ≤ n). The answer to the i-th query is the length of the maximum correct bracket subsequence of sequence sli, sli + 1, ..., sri. Help Sereja answer all queries.

You can find the definitions for a subsequence and a correct bracket sequence in the notes.

Input

The first line contains a sequence of characters s1, s2, ..., sn (1 ≤ n ≤ 106) without any spaces. Each character is either a \"(\" or a \")\". The second line contains integer m (1 ≤ m ≤ 105) — the number of queries. Each of the next m lines contains a pair of integers. The i-th line contains integers li, ri (1 ≤ li ≤ ri ≤ n) — the description of the i-th query.

Output

Print the answer to each question on a single line. Print the answers in the order they go in the input.

Examples

Input

())(())(())(
7
1 1
2 3
1 2
1 12
8 12
5 11
2 10


Output

0
0
2
10
4
6
6

Note

A subsequence of length |x| of string s = s1s2... s|s| (where |s| is the length of string s) is string x = sk1sk2... sk|x| (1 ≤ k1 < k2 < ... < k|x| ≤ |s|).

A correct bracket sequence is a bracket sequence that can be transformed into a correct aryphmetic expression by inserting characters \"1\" and \"+\" between the characters of the string. For example, bracket sequences \"()()\", \"(())\" are correct (the resulting expressions \"(1)+(1)\", \"((1+1)+1)\"), and \")(\" and \"(\" are not.

For the third query required sequence will be «()».

For the fourth query required sequence will be «()(())(())».

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
string str;
struct node_info {
  int c, f, s;
} var;
node_info tree[4000001];
inline void build(int node, int start, int end) {
  if (start == end) {
    tree[node].c = 0;
    if (str[start] == '(') {
      tree[node].f = 1;
      tree[node].s = 0;
    } else {
      tree[node].f = 0;
      tree[node].s = 1;
    }
  } else {
    int mid = (start + end) / 2;
    build(2 * node + 1, start, mid);
    build(2 * node + 2, mid + 1, end);
    int extra = min(tree[2 * node + 1].f, tree[2 * node + 2].s);
    tree[node].c = tree[2 * node + 1].c + tree[2 * node + 2].c + extra;
    tree[node].f = tree[2 * node + 1].f + tree[2 * node + 2].f - extra;
    tree[node].s = tree[2 * node + 1].s + tree[2 * node + 2].s - extra;
  }
}
inline node_info query(int node, int start, int end, int l, int r) {
  if (start > end || start > r || end < l) {
    var.c = var.f = var.s = 0;
    return var;
  }
  if (start >= l && end <= r) return tree[node];
  int mid = (start + end) / 2;
  node_info p1 = query(2 * node + 1, start, mid, l, r);
  node_info p2 = query(2 * node + 2, mid + 1, end, l, r);
  int ans = p1.c + p2.c + min(p1.f, p2.s);
  node_info ret;
  ret.c = p1.c + p2.c + min(p1.f, p2.s);
  ret.f = p1.f + p2.f - min(p1.f, p2.s);
  ret.s = p1.s + p2.s - min(p1.f, p2.s);
  return ret;
}
int main() {
  cin >> str;
  build(0, 0, str.length() - 1);
  int q, l, r;
  scanf(\"%d\", &q);
  while (q--) {
    scanf(\"%d%d\", &l, &r);
    int res = (query(0, 0, str.length() - 1, --l, --r).c) * 2;
    printf(\"%d\n\", res);
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

class Eserejaandbracketsbootcamp(Basebootcamp):
    def __init__(self, n=10, m=3):
        """
        初始化括号查询训练场，设置默认的字符串长度n和查询数量m。
        """
        self.n = n
        self.m = m

    def case_generator(self):
        """
        生成一个随机的括号字符串和相应的查询区间。
        """
        s = ''.join(random.choice(['(', ')']) for _ in range(self.n))
        queries = []
        for _ in range(self.m):
            li = random.randint(1, self.n)
            ri = random.randint(li, self.n)
            queries.append([li, ri])
        return {
            "s": s,
            "queries": queries
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        queries = question_case['queries']
        m = len(queries)
        input_lines = [s, str(m)]
        input_lines.extend(f"{l} {r}" for l, r in queries)
        input_str = '\n'.join(input_lines)
        prompt = f"""Sereja有一个由括号组成的字符串s，其长度为{len(s)}。他需要处理{m}个查询，每个查询要求找出特定区间内的最长有效括号子序列的长度。有效括号子序列是指可以通过插入数字和运算符形成合法算术表达式的括号序列。

例如，输入：
())(())(())(
7
1 1
2 3
...（其余查询）

对应的输出为每个查询的答案，每个答案占一行。

请解决以下问题：
输入：
{input_str}

请按照顺序输出每个查询的结果，每个结果占一行，并将所有答案放在[answer]和[/answer]标签之间。例如：
[answer]
0
0
2
...
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        answers = []
        for line in last_answer.split('\n'):
            line = line.strip()
            if line:
                try:
                    answers.append(int(line))
                except ValueError:
                    pass
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list):
            return False
        s = identity['s']
        queries = identity['queries']
        if len(solution) != len(queries):
            return False

        for idx, (l, r) in enumerate(queries):
            l0 = l - 1
            r0 = r - 1
            if l0 < 0 or r0 >= len(s):
                correct = 0
            else:
                c, _, _ = compute_query(s, l0, r0)
                correct = c * 2
            if solution[idx] != correct:
                return False
        return True

def compute_query(s, l, r):
    if l > r:
        return (0, 0, 0)
    if l == r:
        if s[l] == '(':
            return (0, 1, 0)
        else:
            return (0, 0, 1)
    mid = (l + r) // 2
    left_c, left_f, left_s = compute_query(s, l, mid)
    right_c, right_f, right_s = compute_query(s, mid + 1, r)
    extra = min(left_f, right_s)
    c = left_c + right_c + extra
    f = left_f + right_f - extra
    s_total = left_s + right_s - extra
    return (c, f, s_total)
