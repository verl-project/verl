"""# 

### 谜题描述
John Doe decided that some mathematical object must be named after him. So he invented the Doe graphs. The Doe graphs are a family of undirected graphs, each of them is characterized by a single non-negative number — its order. 

We'll denote a graph of order k as D(k), and we'll denote the number of vertices in the graph D(k) as |D(k)|. Then let's define the Doe graphs as follows:

  * D(0) consists of a single vertex, that has number 1. 
  * D(1) consists of two vertices with numbers 1 and 2, connected by an edge. 
  * D(n) for n ≥ 2 is obtained from graphs D(n - 1) and D(n - 2). D(n - 1) and D(n - 2) are joined in one graph, at that numbers of all vertices of graph D(n - 2) increase by |D(n - 1)| (for example, vertex number 1 of graph D(n - 2) becomes vertex number 1 + |D(n - 1)|). After that two edges are added to the graph: the first one goes between vertices with numbers |D(n - 1)| and |D(n - 1)| + 1, the second one goes between vertices with numbers |D(n - 1)| + 1 and 1. Note that the definition of graph D(n) implies, that D(n) is a connected graph, its vertices are numbered from 1 to |D(n)|. 

<image> The picture shows the Doe graphs of order 1, 2, 3 and 4, from left to right.

John thinks that Doe graphs are that great because for them exists a polynomial algorithm for the search of Hamiltonian path. However, your task is to answer queries of finding the shortest-length path between the vertices ai and bi in the graph D(n).

A path between a pair of vertices u and v in the graph is a sequence of vertices x1, x2, ..., xk (k > 1) such, that x1 = u, xk = v, and for any i (i < k) vertices xi and xi + 1 are connected by a graph edge. The length of path x1, x2, ..., xk is number (k - 1).

Input

The first line contains two integers t and n (1 ≤ t ≤ 105; 1 ≤ n ≤ 103) — the number of queries and the order of the given graph. The i-th of the next t lines contains two integers ai and bi (1 ≤ ai, bi ≤ 1016, ai ≠ bi) — numbers of two vertices in the i-th query. It is guaranteed that ai, bi ≤ |D(n)|.

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use cin, cout streams or the %I64d specifier. 

Output

For each query print a single integer on a single line — the length of the shortest path between vertices ai and bi. Print the answers to the queries in the order, in which the queries are given in the input.

Examples

Input

10 5
1 2
1 3
1 4
1 5
2 3
2 4
2 5
3 4
3 5
4 5


Output

1
1
1
2
1
2
3
1
2
1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long f[80], d[80];
pair<long long, int> p1[80][2], p2[80][2];
int T, n;
inline void init(long long v, int n, int id) {
  if (n == 1) {
    p1[0][id] = make_pair(1, 0);
    p2[0][id] = make_pair(1, 0);
    p1[1][id] = make_pair(v, v != 1);
    p2[1][id] = make_pair(v, v != 2);
    return;
  }
  if (v > f[n - 1])
    init(v - f[n - 1], n - 1, id);
  else
    init(v, n - 1, id);
  p1[n][id].first = p2[n][id].first = v;
  if (v > f[n - 1])
    p1[n][id].second = p1[n - 2][id].second + 1;
  else
    p1[n][id].second = min(p2[n - 1][id].second + 2, p1[n - 1][id].second);
  if (v > f[n - 1])
    p2[n][id].second = p2[n - 2][id].second;
  else
    p2[n][id].second =
        d[n - 2] + 1 + min(p2[n - 1][id].second, p1[n - 1][id].second);
}
inline int query(long long a, long long b, int n) {
  if (a > b) swap(a, b);
  if (a == b) return 0;
  if (n == 0) return 0;
  if (n == 1) return a != b;
  if (a == 1) {
    if (b == p1[n][0].first) return p1[n][0].second;
    if (b == p1[n][1].first) return p1[n][1].second;
  }
  if (b == f[n]) {
    if (a == p2[n][0].first) return p2[n][0].second;
    if (a == p2[n][1].first) return p2[n][1].second;
  }
  int ret = 0x3f3f3f3f;
  if (a > f[n - 1] && b > f[n - 1])
    ret = query(a - f[n - 1], b - f[n - 1], n - 2);
  else if (b > f[n - 1])
    ret = query(1, b - f[n - 1], n - 2) +
          min(query(1, a, n - 1), query(a, f[n - 1], n - 1)) + 1;
  else
    ret = min(min(query(a, f[n - 1], n - 1) + query(1, b, n - 1),
                  query(a, 1, n - 1) + query(b, f[n - 1], n - 1)) +
                  2,
              query(a, b, n - 1));
  return ret;
}
int main() {
  f[0] = 1;
  f[1] = 2;
  for (int i = 2; i <= 77; i++) f[i] = f[i - 1] + f[i - 2];
  d[0] = 0;
  d[1] = 1;
  for (int i = 2; i <= 77; i++) d[i] = d[i - 2] + 1;
  scanf(\"%d%d\", &T, &n);
  if (n > 77) n = 78;
  while (T--) {
    long long x, y;
    scanf(\"%lld%lld\", &x, &y);
    init(x, n, 0);
    init(y, n, 1);
    printf(\"%d\n\", query(x, y, n));
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from functools import lru_cache
from bootcamp import Basebootcamp
import re

class Edoegraphsbootcamp(Basebootcamp):
    _f = [1, 2]  # 类属性，预计算斐波那契数列

    def __init__(self, max_n=20, max_queries=10):
        self.max_n = max_n
        self.max_queries = max_queries

    @classmethod
    def _ensure_f(cls, n):
        while len(cls._f) <= n:
            next_val = cls._f[-1] + cls._f[-2]
            cls._f.append(next_val)

    @staticmethod
    @lru_cache(maxsize=None)
    def compute_shortest_path(a, b, n):
        if a == b:
            return 0
        if a > b:
            a, b = b, a
        if n == 0:
            return 0
        if n == 1:
            return 1
        
        Edoegraphsbootcamp._ensure_f(n - 1)
        fn_minus_1 = Edoegraphsbootcamp._f[n - 1]
        a_in_B = a > fn_minus_1
        b_in_B = b > fn_minus_1
        
        if a_in_B and b_in_B:
            new_a = a - fn_minus_1
            new_b = b - fn_minus_1
            return Edoegraphsbootcamp.compute_shortest_path(new_a, new_b, n - 2)
        elif b_in_B:
            part_b = Edoegraphsbootcamp.compute_shortest_path(1, b - fn_minus_1, n - 2)
            option1 = Edoegraphsbootcamp.compute_shortest_path(a, fn_minus_1, n - 1)
            option2 = Edoegraphsbootcamp.compute_shortest_path(a, 1, n - 1)
            part_a = min(option1, option2)
            return part_a + part_b + 1
        else:
            option1 = Edoegraphsbootcamp.compute_shortest_path(a, b, n - 1)
            optionA = Edoegraphsbootcamp.compute_shortest_path(a, fn_minus_1, n - 1) + \
                      Edoegraphsbootcamp.compute_shortest_path(1, b, n - 1) + 2
            optionB = Edoegraphsbootcamp.compute_shortest_path(a, 1, n - 1) + \
                      Edoegraphsbootcamp.compute_shortest_path(fn_minus_1, b, n - 1) + 2
            option2 = min(optionA, optionB)
            return min(option1, option2)

    def case_generator(self):
        n = random.randint(1, self.max_n)
        self._ensure_f(n)
        fn = self._f[n]
        queries = []
        for _ in range(self.max_queries):
            a = random.randint(1, fn)
            b = random.randint(1, fn)
            while a == b:
                b = random.randint(1, fn)
            queries.append((a, b))
        return {'n': n, 'queries': queries, 'f_n': fn}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        queries = question_case['queries']
        prompt = f"""你是图论专家，请解决以下Doe图的最短路径问题。

Doe图D(k)根据其阶数k构建。D(0)是单个顶点1；D(1)是两个顶点1和2的边。对于k≥2，D(k)由D(k-1)和D(k-2)合并，并添加两条边连接他们的顶点。每个顶点都有唯一的编号。

给定阶数为{n}的Doe图D({n})，顶点数目为{question_case['f_n']}。现有{len(queries)}个查询，每个查询要求计算两个不同顶点之间的最短路径长度。

输入格式：每个查询给出两个顶点编号a_i和b_i。

输出格式：对于每个查询，输出一个整数，表示最短路径的长度。

请按顺序对每个查询给出答案，每个答案占一行，并将所有答案放在[answer]和[/answer]之间。

例如，对于两个查询的输出应为：

[answer]
3
5
[/answer]

现在，处理以下查询：
"""
        for i, (a, b) in enumerate(queries, 1):
            prompt += f"查询{i}: {a} {b}\n"
        prompt += "\n请按要求输出答案。"
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_block = matches[-1].strip()
        answers = []
        for line in answer_block.split('\n'):
            line = line.strip()
            if line:
                if line.isdigit() or (line.startswith('-') and line[1:].isdigit()):
                    answers.append(int(line))
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        n = identity['n']
        queries = identity['queries']
        if len(solution) != len(queries):
            return False
        for (a, b), user_ans in zip(queries, solution):
            correct_ans = cls.compute_shortest_path(a, b, n)
            if user_ans != correct_ans:
                return False
        return True
