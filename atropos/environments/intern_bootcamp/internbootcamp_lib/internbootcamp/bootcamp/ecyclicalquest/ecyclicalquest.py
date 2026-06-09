"""# 

### 谜题描述
Some days ago, WJMZBMR learned how to answer the query \"how many times does a string x occur in a string s\" quickly by preprocessing the string s. But now he wants to make it harder.

So he wants to ask \"how many consecutive substrings of s are cyclical isomorphic to a given string x\". You are given string s and n strings xi, for each string xi find, how many consecutive substrings of s are cyclical isomorphic to xi.

Two strings are called cyclical isomorphic if one can rotate one string to get the other one. 'Rotate' here means 'to take some consecutive chars (maybe none) from the beginning of a string and put them back at the end of the string in the same order'. For example, string \"abcde\" can be rotated to string \"deabc\". We can take characters \"abc\" from the beginning and put them at the end of \"de\".

Input

The first line contains a non-empty string s. The length of string s is not greater than 106 characters.

The second line contains an integer n (1 ≤ n ≤ 105) — the number of queries. Then n lines follow: the i-th line contains the string xi — the string for the i-th query. The total length of xi is less than or equal to 106 characters.

In this problem, strings only consist of lowercase English letters.

Output

For each query xi print a single integer that shows how many consecutive substrings of s are cyclical isomorphic to xi. Print the answers to the queries in the order they are given in the input.

Examples

Input

baabaabaaa
5
a
ba
baa
aabaa
aaba


Output

7
5
7
3
5


Input

aabbaa
3
aa
aabb
abba


Output

2
3
3

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = (int)1e6 + 10, mod = (int)1e9 + 7;
int q, last, sz;
char s[N];
struct state {
  int len, link, next[26], cnt;
};
vector<state> st;
vector<bool> vis;
vector<int> adj[2 * N];
void sa_init() {
  st.resize(2 * N);
  vis.resize(2 * N);
  st[0].link = -1;
  st[0].len = 0;
  last = 0;
  sz = 1;
}
void sa_extend(char ch) {
  int c = ch - 'a';
  int cur = sz++;
  st[cur].len = st[last].len + 1;
  int p = last;
  while (p != -1 && !st[p].next[c]) {
    st[p].next[c] = cur;
    p = st[p].link;
  }
  if (p == -1)
    st[cur].link = 0;
  else {
    int q = st[p].next[c];
    if (st[q].len == st[p].len + 1)
      st[cur].link = q;
    else {
      int clone = sz++;
      st[clone] = st[q];
      st[clone].cnt = 0;
      st[clone].len = st[p].len + 1;
      while (p != -1 && st[p].next[c] == q) {
        st[p].next[c] = clone;
        p = st[p].link;
      }
      st[q].link = st[cur].link = clone;
    }
  }
  last = cur;
  st[cur].cnt = 1;
}
void dfs(int u) {
  for (int v : adj[u]) {
    dfs(v);
    st[u].cnt += st[v].cnt;
  }
}
int main() {
  sa_init();
  char x;
  while ((x = getchar()) != '\n') sa_extend(x);
  for (int i = 1; i < sz; i++) adj[st[i].link].push_back(i);
  dfs(0);
  scanf(\"%d\", &q);
  for (int i = 0; i < q; i++) {
    scanf(\"%s\", &s);
    vector<int> v;
    int n = strlen(s);
    int l = 0, len = 0, u = 0, ans = 0;
    for (int j = 0; j < n; j++) {
      while (len < n) {
        int c = s[l] - 'a';
        if (!st[u].next[c]) break;
        u = st[u].next[c];
        l++, len++;
        if (l == n) l = 0;
      }
      if (len == n) {
        if (!vis[u]) ans += st[u].cnt;
        vis[u] = 1;
        v.push_back(u);
      }
      if (u) {
        if (st[st[u].link].len == len - 1) {
          u = st[u].link;
        }
        len--;
      } else
        l = (l + 1) % n;
    }
    for (int a : v) vis[a] = 0;
    printf(\"%d\n\", ans);
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

def minimal_representation(s):
    n = len(s)
    if n == 0:
        return ''
    s += s
    i, j, k = 0, 1, 0
    while i < n and j < n and k < n:
        if s[i + k] == s[j + k]:
            k += 1
        else:
            if s[i + k] > s[j + k]:
                i += k + 1
            else:
                j += k + 1
            if i == j:
                j += 1
            k = 0
    min_pos = min(i, j)
    return s[min_pos:min_pos + n]

class Ecyclicalquestbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_s_length = params.get('max_s_length', 10)
        self.min_s_length = params.get('min_s_length', 5)
        self.max_queries = params.get('max_queries', 5)
        self.min_queries = params.get('min_queries', 1)
    
    def case_generator(self):
        def generate_random_string(length):
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        
        s_length = random.randint(self.min_s_length, self.max_s_length)
        s = generate_random_string(s_length)
        n = random.randint(self.min_queries, self.max_queries)
        queries = [generate_random_string(random.randint(1, s_length + 2)) for _ in range(n)]
        
        answers = []
        for xi in queries:
            xi_len = len(xi)
            if xi_len > len(s):
                answers.append(0)
                continue
            min_xi = minimal_representation(xi)
            count = 0
            substr_len = xi_len
            s_len = len(s)
            for i in range(s_len - substr_len + 1):
                substr = s[i:i + substr_len]
                if minimal_representation(substr) == min_xi:
                    count += 1
            answers.append(count)
        
        return {
            's': s,
            'n': n,
            'queries': queries,
            'answers': answers
        }
    
    @staticmethod
    def prompt_func(question_case):
        s = question_case['s']
        n = question_case['n']
        queries = question_case['queries']
        problem = (
            "给定一个字符串 s 和若干查询字符串，每个查询要求计算 s 中连续子串与查询字符串循环同构的数量。"
            "两个字符串循环同构当且仅当其中一个可以通过旋转得到另一个（旋转指将前k个字符移到末尾）。\n\n"
            f"输入格式：\n第一行：{s}\n第二行：{n}\n接下来 {n} 行每行一个查询字符串：\n"
        )
        for xi in queries:
            problem += f"{xi}\n"
        problem += (
            "\n请按顺序输出每个查询的结果，每个结果占一行，置于[answer]和[/answer]之间。例如：\n"
            "[answer]\n3\n0\n5\n[/answer]\n"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        numbers = []
        for line in last_match.splitlines():
            stripped = line.strip()
            if stripped:
                try:
                    numbers.append(int(stripped))
                except ValueError:
                    continue
        return numbers if numbers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answers']
