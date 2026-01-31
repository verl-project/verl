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
const int N = 100010;
const int inf = 0x3f3f3f3f;
using namespace std;
int n, t;
long long a, b, f[N];
int dp1[110][2], dp2[110][2];
int dfs1(int a, int b, long long c) {
  if (a == 1) return c + b == 2;
  if (a == 0) return 0;
  int &ret = dp1[a][b];
  if (ret + 1) return dp1[a][b];
  if (b) {
    if (c > f[a - 1])
      ret = dfs1(a - 2, 1, c - f[a - 1]);
    else
      ret = min(dfs1(a - 1, 1, c), dfs1(a - 1, 0, c)) + 1 + (a - 1) / 2;
  } else {
    if (c > f[a - 1])
      ret = dfs1(a - 2, 0, c - f[a - 1]) + 1;
    else
      ret = min(dfs1(a - 1, 0, c), dfs1(a - 1, 1, c) + 2);
  }
  return ret;
}
int dfs2(int a, int b, long long c) {
  if (a == 1) return c + b == 2;
  if (a == 0) return 0;
  int &ret = dp2[a][b];
  if (ret + 1) return ret;
  if (b) {
    if (c > f[a - 1])
      ret = dfs2(a - 2, 1, c - f[a - 1]);
    else
      ret = min(dfs2(a - 1, 1, c), dfs2(a - 1, 0, c)) + 1 + (a - 1) / 2;
  } else {
    if (c > f[a - 1])
      ret = dfs2(a - 2, 0, c - f[a - 1]) + 1;
    else
      ret = min(dfs2(a - 1, 0, c), dfs2(a - 1, 1, c) + 2);
  }
  return ret;
}
int dfs(long long a, long long b, int k) {
  if (a == b) return 0;
  if (k == 1) return 1;
  if (a > f[k - 1] && b > f[k - 1])
    return dfs(a - f[k - 1], b - f[k - 1], k - 2);
  if (a <= f[k - 1] && b <= f[k - 1]) {
    int ret = dfs(a, b, k - 1);
    ret = min(ret, dfs1(k - 1, 0, a) + dfs2(k - 1, 1, b) + 2);
    ret = min(ret, dfs1(k - 1, 1, a) + dfs2(k - 1, 0, b) + 2);
    return ret;
  }
  return min(dfs1(k - 1, 1, a), dfs1(k - 1, 0, a)) +
         dfs2(k - 2, 0, b - f[k - 1]) + 1;
}
int main() {
  cin >> t >> n;
  if (n > 90) n = 90;
  f[0] = 1;
  f[1] = 2;
  for (int i = 2; i <= 91; i++) f[i] = f[i - 1] + f[i - 2];
  for (int i = 0; i < t; i++) {
    cin >> a >> b;
    memset(dp1, -1, sizeof(dp1));
    memset(dp2, -1, sizeof(dp2));
    if (a > b) swap(a, b);
    cout << dfs(a, b, n) << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random
from functools import lru_cache

class Cdoegraphsbootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_queries=10):
        self.max_n = min(max_n, 90)  # Clamp max_n to 90 as per reference code
        self.max_queries = max_queries
    
    def case_generator(self):
        # Generate a random order n within the allowed range (1 to max_n)
        n = random.randint(1, self.max_n)
        # Compute the size of D(n) using the correct Fibonacci sequence
        fib = self.compute_doe_fib(n)
        d_size = fib[-1]
        # Generate t random valid queries
        t = random.randint(1, self.max_queries)
        queries = []
        for _ in range(t):
            a = random.randint(1, d_size)
            b = random.randint(1, d_size)
            while a == b:
                b = random.randint(1, d_size)
            a, b = sorted((a, b))
            queries.append((a, b))
        return {
            'n': n,
            'queries': queries,
            'd_size': d_size
        }
    
    @staticmethod
    def compute_doe_fib(n):
        """Generates the Fibonacci sequence for Doe graph sizes up to order n (0-based)."""
        if n < 0:
            return []
        fib = [1]  # D(0)
        if n == 0:
            return fib
        fib.append(2)  # D(1)
        for i in range(2, n + 1):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    @staticmethod
    def prompt_func(question_case):
        prompt = (
            "Solve the shortest path problem in a Doe graph D(n).\n"
            "Rules:\n"
            "- D(0): 1 vertex (1).\n"
            "- D(1): 2 vertices (1-2) connected.\n"
            "- D(n) for n ≥ 2 combines D(n-1) and D(n-2). Vertices in D(n-2) are renumbered by adding |D(n-1)|.\n"
            "Two new edges connect |D(n-1)| to |D(n-1)|+1 and |D(n-1)|+1 to 1.\n\n"
            f"Given D({question_case['n']}) with {len(question_case['queries'])} queries, provide the shortest path length for each pair.\n"
            "Queries:\n"
        )
        for i, (a, b) in enumerate(question_case['queries'], 1):
            prompt += f"{i}. {a} ↔ {b}\n"
        prompt += (
            "\nEnclose answers within [answer] tags, each on separate lines.\n"
            "Example:\n[answer]\n3\n2\n5\n[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        # Extract the last answer block and parse all integers
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        answer_block = answer_blocks[-1]
        # Extract all integers in the block
        answers = list(map(int, re.findall(r'\b\d+\b', answer_block)))
        return answers if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != len(identity['queries']):
            return False
        
        n = identity['n']
        queries = identity['queries']
        limited_n = min(n, 90)  # Reference code limits to 90
        fib = cls.compute_doe_fib(limited_n)
        if len(fib) < limited_n + 1:
            return False  # Invalid Fibonacci sequence
        
        # Convert to tuple for memoization hashability
        fib_tuple = tuple(fib)
        
        for i, (a, b) in enumerate(queries):
            try:
                correct = cls.dfs(a, b, limited_n, fib_tuple)
                if solution[i] != correct:
                    return False
            except:
                return False
        return True
    
    @classmethod
    def dfs(cls, a, b, k, fib_tuple):
        if a == b:
            return 0
        if k == 1:
            return 1
        if a > b:
            a, b = b, a
        return cls._dfs(a, b, k, fib_tuple)
    
    @classmethod
    def _dfs(cls, a, b, k, fib_tuple):
        if a == b:
            return 0
        if k == 1:
            return 1
        if k == 0:
            return 0
        
        size_k_1 = fib_tuple[k-1]
        if a > size_k_1 and b > size_k_1:
            return cls._dfs(a - size_k_1, b - size_k_1, k-2, fib_tuple)
        if a <= size_k_1 and b <= size_k_1:
            path_in = cls._dfs(a, b, k-1, fib_tuple)
            path1 = cls.dfs1(k-1, 0, a, fib_tuple) + cls.dfs2(k-1, 1, b, fib_tuple) + 2
            path2 = cls.dfs1(k-1, 1, a, fib_tuple) + cls.dfs2(k-1, 0, b, fib_tuple) + 2
            return min(path_in, path1, path2)
        else:
            path1 = min(cls.dfs1(k-1, 0, a, fib_tuple), cls.dfs1(k-1, 1, a, fib_tuple))
            path2 = cls.dfs2(k-2, 0, b - size_k_1, fib_tuple) + 1
            return path1 + path2
    
    @classmethod
    @lru_cache(maxsize=None)
    def dfs1(cls, a, b, c, fib_tuple):
        if a == 1:
            return 1 if (c + b) == 2 else 0
        if a == 0:
            return 0
        size_a_1 = fib_tuple[a-1]
        if b:
            if c > size_a_1:
                return cls.dfs1(a-2, 1, c - size_a_1, fib_tuple)
            else:
                option1 = cls.dfs1(a-1, 1, c, fib_tuple)
                option2 = cls.dfs1(a-1, 0, c, fib_tuple)
                return min(option1, option2) + 1 + (a-1) // 2
        else:
            if c > size_a_1:
                return cls.dfs1(a-2, 0, c - size_a_1, fib_tuple) + 1
            else:
                option1 = cls.dfs1(a-1, 0, c, fib_tuple)
                option2 = cls.dfs1(a-1, 1, c, fib_tuple) + 2
                return min(option1, option2)
    
    @classmethod
    @lru_cache(maxsize=None)
    def dfs2(cls, a, b, c, fib_tuple):
        if a == 1:
            return 1 if (c + b) == 2 else 0
        if a == 0:
            return 0
        size_a_1 = fib_tuple[a-1]
        if b:
            if c > size_a_1:
                return cls.dfs2(a-2, 1, c - size_a_1, fib_tuple)
            else:
                option1 = cls.dfs2(a-1, 1, c, fib_tuple)
                option2 = cls.dfs2(a-1, 0, c, fib_tuple)
                return min(option1, option2) + 1 + (a-1) // 2
        else:
            if c > size_a_1:
                return cls.dfs2(a-2, 0, c - size_a_1, fib_tuple) + 1
            else:
                option1 = cls.dfs2(a-1, 0, c, fib_tuple)
                option2 = cls.dfs2(a-1, 1, c, fib_tuple) + 2
                return min(option1, option2)
