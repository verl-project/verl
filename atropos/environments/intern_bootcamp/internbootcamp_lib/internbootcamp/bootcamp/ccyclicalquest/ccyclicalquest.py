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
void io() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  cout.precision(15);
}
const int inf = 1e9;
const int maxn = 4 * 1e6 + 5;
char s[maxn];
unordered_map<int, int> to[maxn];
int len[maxn], fipos[maxn], link[maxn];
int depth[maxn];
int node, pos;
int sz = 1, n = 0;
int remain = 0;
int make_node(int _pos, int _len) {
  fipos[sz] = _pos;
  len[sz] = _len;
  return sz++;
}
void go_edge(int &node, int &pos) {
  while (pos > len[to[node][s[n - pos]]]) {
    node = to[node][s[n - pos]];
    pos -= len[node];
  }
}
void add_letter(int c) {
  s[n++] = c;
  pos++;
  int last = 0;
  remain++;
  while (pos > 0) {
    go_edge(node, pos);
    int edge = s[n - pos];
    int &v = to[node][edge];
    int t = s[fipos[v] + pos - 1];
    if (v == 0) {
      remain--;
      v = make_node(n - pos, inf);
      link[last] = node;
      last = 0;
    } else if (t == c) {
      link[last] = node;
      return;
    } else {
      remain--;
      int u = make_node(fipos[v], pos - 1);
      to[u][c] = make_node(n - 1, inf);
      to[u][t] = v;
      fipos[v] += pos - 1;
      len[v] -= pos - 1;
      v = u;
      link[last] = u;
      last = u;
    }
    if (node == 0)
      pos--;
    else
      node = link[node];
  }
}
int val[maxn];
int slen;
void dfs(int u, int d = 0) {
  val[u] = (len[u] > slen + 2) && (u != 0);
  depth[u] = min(inf, d);
  for (auto it : to[u]) {
    int v = it.second;
    if (v) {
      dfs(v, d + len[v]);
      val[u] += val[v];
    }
  }
}
void fix(int &node, int &edg, int &pos, int &st, int &en, string &s) {
  int fichar = s[st + depth[node]];
  edg = to[node][fichar];
  while (edg && pos > len[edg]) {
    node = edg;
    pos -= len[edg];
    fichar = s[st + depth[node]];
    edg = to[node][fichar];
  }
}
int main(int argc, char *argv[]) {
  io();
  len[0] = inf;
  string s;
  cin >> s;
  slen = s.length();
  s += '$';
  for (auto c : s) {
    add_letter(c);
  }
  dfs(0);
  int q;
  cin >> q;
  string str;
  while (q--) {
    cin >> str;
    int l = str.length();
    if (l > slen) {
      cout << 0 << '\n';
      continue;
    }
    str = str + str;
    long long ans = 0;
    int st = 0;
    int en = 0;
    int node = 0;
    int edg = 0;
    int pos = 1;
    set<pair<int, pair<int, int>>> done;
    while (st < l) {
      fix(node, edg, pos, st, en, str);
      assert(st <= en);
      if (edg == 0) {
        if (node == 0) {
          pos--;
        } else {
          node = link[node];
        }
        st++;
        if (st > en) {
          en++;
        }
        continue;
      }
      while (en < (st + l)) {
        fix(node, edg, pos, st, en, str);
        int spos = fipos[edg] + pos - 1;
        if (spos >= slen) {
          break;
        }
        if (s[spos] == str[st + depth[node] + pos - 1]) {
          en++;
          pos++;
          fix(node, edg, pos, st, en, str);
        } else {
          break;
        }
      }
      if (en == st + l) {
        if (done.find({node, {edg, pos}}) == done.end()) {
          done.insert({node, {edg, pos}});
          if (pos == 1) {
            ans = ans + val[node];
          } else {
            if (edg != 0) ans = ans + val[edg];
          }
        } else {
        }
      }
      st++;
      if (st > en) {
        en++;
      }
      if (node == 0)
        pos--;
      else
        node = link[node];
    }
    cout << ans << '\n';
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
import re
from bootcamp import Basebootcamp

class Ccyclicalquestbootcamp(Basebootcamp):
    def __init__(self, s_length=10, num_queries=5, query_length_min=1, query_length_max=10, seed=None):
        """
        初始化训练场，设置生成谜题实例的参数。
        
        参数:
            s_length: 字符串 s 的长度，默认为 10。
            num_queries: 查询的数量，默认为 5。
            query_length_min: 查询字符串的最小长度，默认为 1。
            query_length_max: 查询字符串的最大长度，默认为 10。
            seed: 随机种子，用于生成可重复的结果，默认为 None。
        """
        self.s_length = s_length
        self.num_queries = num_queries
        self.query_length_min = query_length_min
        self.query_length_max = query_length_max
        self.seed = seed
        
    def case_generator(self):
        """
        生成谜题实例，包括字符串 s 和若干查询字符串 xi，并预先计算每个查询的正确答案。
        
        返回:
            一个包含 's', 'queries', 和 'answers' 的字典。
        """
        random.seed(self.seed)
        s = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=self.s_length))
        queries = []
        for _ in range(self.num_queries):
            # 确保查询字符串的长度不超过 s 的长度
            m = random.randint(self.query_length_min, min(self.query_length_max, self.s_length))
            xi = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=m))
            queries.append(xi)
        
        answers = []
        for xi in queries:
            m = len(xi)
            if m == 0 or m > self.s_length:
                answers.append(0)
                continue
            # 生成所有可能的循环旋转形式
            rotations = set()
            for i in range(m):
                rotated = xi[i:] + xi[:i]
                rotations.add(rotated)
            # 统计这些旋转形式在 s 中的出现次数
            count = 0
            for rot in rotations:
                rot_len = len(rot)
                for i in range(len(s) - rot_len + 1):
                    if s[i:i+rot_len] == rot:
                        count += 1
            answers.append(count)
        
        case = {
            's': s,
            'queries': queries,
            'answers': answers
        }
        return case
    
    @staticmethod
    def prompt_func(question_case):
        """
        将生成的谜题实例转换为文本形式的问题。
        
        参数:
            question_case: 由 case_generator 生成的谜题实例。
            
        返回:
            格式化的问题字符串。
        """
        s = question_case['s']
        queries = question_case['queries']
        prompt = f"给定字符串 s='{s}'，请回答以下查询：对于每个查询字符串 xi，计算它在 s 中有多少个循环同构的子串。循环同构的定义是两个字符串可以通过旋转得到对方。例如，'abcde' 可以旋转为 'deabc'。请将每个查询的结果按顺序排列，并将答案放在 [answer] 标签中。\n"
        for i, xi in enumerate(queries, 1):
            prompt += f"查询 {i}：'{xi}'\n"
        prompt += "请将答案按顺序排列，放在 [answer] 标签中，例如：\n[answer]7 5 7 3 5[/answer]"
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        从模型的回复中提取符合格式要求的答案。
        
        参数:
            output: 模型的完整输出。
            
        返回:
            提取的答案列表，如果未找到符合格式的答案则返回 None。
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            answers = list(map(int, last_match.split()))
            return answers
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证提取的答案是否正确。
        
        参数:
            solution: extract_output 提取的答案。
            identity: case_generator 生成的谜题实例。
            
        返回:
            bool: 答案是否正确。
        """
        if solution is None:
            return False
        expected = identity['answers']
        if len(solution) != len(expected):
            return False
        for s_ans, e_ans in zip(solution, expected):
            if s_ans != e_ans:
                return False
        return True
