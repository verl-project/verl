"""# 

### 谜题描述
You are given two permutations p and q, consisting of n elements, and m queries of the form: l1, r1, l2, r2 (l1 ≤ r1; l2 ≤ r2). The response for the query is the number of such integers from 1 to n, that their position in the first permutation is in segment [l1, r1] (borders included), and position in the second permutation is in segment [l2, r2] (borders included too).

A permutation of n elements is the sequence of n distinct integers, each not less than 1 and not greater than n.

Position of number v (1 ≤ v ≤ n) in permutation g1, g2, ..., gn is such number i, that gi = v.

Input

The first line contains one integer n (1 ≤ n ≤ 106), the number of elements in both permutations. The following line contains n integers, separated with spaces: p1, p2, ..., pn (1 ≤ pi ≤ n). These are elements of the first permutation. The next line contains the second permutation q1, q2, ..., qn in same format.

The following line contains an integer m (1 ≤ m ≤ 2·105), that is the number of queries.

The following m lines contain descriptions of queries one in a line. The description of the i-th query consists of four integers: a, b, c, d (1 ≤ a, b, c, d ≤ n). Query parameters l1, r1, l2, r2 are obtained from the numbers a, b, c, d using the following algorithm: 

  1. Introduce variable x. If it is the first query, then the variable equals 0, else it equals the response for the previous query plus one. 
  2. Introduce function f(z) = ((z - 1 + x) mod n) + 1. 
  3. Suppose l1 = min(f(a), f(b)), r1 = max(f(a), f(b)), l2 = min(f(c), f(d)), r2 = max(f(c), f(d)). 

Output

Print a response for each query in a separate line.

Examples

Input

3
3 1 2
3 2 1
1
1 2 3 3


Output

1


Input

4
4 3 2 1
2 3 4 1
3
1 2 3 4
1 3 2 1
1 4 2 3


Output

1
1
2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MOD = 998244353;
struct tnode {
  int sum;
  tnode *lson, *rson;
  tnode(int x = 0) {
    sum = x;
    lson = rson = NULL;
  }
};
void pushup(tnode* cur) {
  cur->sum = (cur->lson == NULL ? 0 : cur->lson->sum) +
             (cur->rson == NULL ? 0 : cur->rson->sum);
}
tnode* modify(tnode* cur, int id, int val, int cl = 0, int cr = 1048575) {
  if (cl == cr) return new tnode(val);
  int mid = (cl + cr) >> 1;
  tnode* ret = new tnode();
  tnode *ls = cur == NULL ? NULL : cur->lson,
        *rs = cur == NULL ? NULL : cur->rson;
  ret->lson = id <= mid ? modify(ls, id, val, cl, mid) : ls;
  ret->rson = id > mid ? modify(rs, id, val, mid + 1, cr) : rs;
  pushup(ret);
  return ret;
}
int query(tnode* cur, int l, int r, int cl = 0, int cr = 1048575) {
  if (cur == NULL) return 0;
  if (l == cl && r == cr) return cur->sum;
  int mid = (cl + cr) >> 1;
  if (r <= mid)
    return query(cur->lson, l, r, cl, mid);
  else if (l > mid)
    return query(cur->rson, l, r, mid + 1, cr);
  else
    return query(cur->lson, l, mid, cl, mid) +
           query(cur->rson, mid + 1, r, mid + 1, cr);
}
int n, q, p0[1000005], occ[1000005], p1[1000005];
tnode* tre[1000005];
int x;
int f(int z) { return (z - 1 + x) % n + 1; }
int main() {
  scanf(\"%d\", &n);
  for (int i = 1; i <= (int)(n); i++) {
    scanf(\"%d\", &p0[i]);
    occ[p0[i]] = i;
  }
  for (int i = 1; i <= (int)(n); i++) scanf(\"%d\", &p1[i]);
  for (int i = 1; i <= (int)(n); i++)
    tre[i] = modify(tre[i - 1], occ[p1[i]], 1);
  scanf(\"%d\", &q);
  x = 0;
  for (int i = 0; i < (int)(q); i++) {
    int a, b, c, d;
    scanf(\"%d%d%d%d\", &a, &b, &c, &d);
    a = f(a);
    b = f(b);
    c = f(c);
    d = f(d);
    if (a > b) swap(a, b);
    if (c > d) swap(c, d);
    x = query(tre[d], a, b) - query(tre[c - 1], a, b) + 1;
    printf(\"%d\n\", x - 1);
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

class Ctwopermutationsbootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_m=5):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        p = list(range(1, n+1))
        random.shuffle(p)
        q = list(range(1, n+1))
        random.shuffle(q)
        m = random.randint(1, self.max_m)
        queries = []
        for _ in range(m):
            a = random.randint(1, n)
            b = random.randint(1, n)
            c = random.randint(1, n)
            d = random.randint(1, n)
            queries.append((a, b, c, d))
        answers = self.calculate_answers(n, p, q, queries)
        return {
            'n': n,
            'p': p,
            'q': q,
            'm': m,
            'queries': queries,
            'answers': answers
        }
    
    @staticmethod
    def calculate_answers(n, p, q, queries):
        pos_p = {v: i+1 for i, v in enumerate(p)}
        pos_q = {v: i+1 for i, v in enumerate(q)}
        x = 0
        answers = []
        for a, b, c, d in queries:
            fa = ((a - 1 + x) % n) + 1
            fb = ((b - 1 + x) % n) + 1
            fc = ((c - 1 + x) % n) + 1
            fd = ((d - 1 + x) % n) + 1
            l1, r1 = sorted([fa, fb])
            l2, r2 = sorted([fc, fd])
            
            valid_p = {v for v in range(1, n+1) if l1 <= pos_p[v] <= r1}
            valid_q = {v for v in range(1, n+1) if l2 <= pos_q[v] <= r2}
            count = len(valid_p & valid_q)
            
            answers.append(count)
            x = count
        return answers
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        p = ' '.join(map(str, question_case['p']))
        q = ' '.join(map(str, question_case['q']))
        m = question_case['m']
        queries = '\n'.join(' '.join(map(str, q)) for q in question_case['queries'])
        prompt = f"""给定两个长度为{n}的排列p和q，处理{m}个查询。每个查询给出参数a,b,c,d，根据以下规则计算：

1. 首个查询x=0，之后x=前一个查询结果
2. 计算f(z)=((z-1 + x) mod n) +1
3. l1=min(f(a),f(b)), r1=max(f(a),f(b))
4. l2=min(f(c),f(d)), r2=max(f(c),f(d))
5. 统计同时满足以下条件的整数v的数量：
   - v在p中的位置位于[l1, r1]区间
   - v在q中的位置位于[l2, r2]区间

输入数据格式：
{n}
{p}
{q}
{m}
{queries}

要求输出每个查询的结果，每个答案单独一行，包裹在[answer]标签内。如：
[answer]
3
0
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        numbers = []
        for line in matches[-1].strip().splitlines():
            cleaned = re.sub(r'\D+', '', line)
            if cleaned:
                numbers.append(int(cleaned))
        return numbers if numbers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answers', [])
