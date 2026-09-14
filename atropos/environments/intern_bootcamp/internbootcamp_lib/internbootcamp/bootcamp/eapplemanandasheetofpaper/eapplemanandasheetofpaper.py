"""# 

### 谜题描述
Appleman has a very big sheet of paper. This sheet has a form of rectangle with dimensions 1 × n. Your task is help Appleman with folding of such a sheet. Actually, you need to perform q queries. Each query will have one of the following types:

  1. Fold the sheet of paper at position pi. After this query the leftmost part of the paper with dimensions 1 × pi must be above the rightmost part of the paper with dimensions 1 × ([current width of sheet] - pi). 
  2. Count what is the total width of the paper pieces, if we will make two described later cuts and consider only the pieces between the cuts. We will make one cut at distance li from the left border of the current sheet of paper and the other at distance ri from the left border of the current sheet of paper. 



Please look at the explanation of the first test example for better understanding of the problem.

Input

The first line contains two integers: n and q (1 ≤ n ≤ 105; 1 ≤ q ≤ 105) — the width of the paper and the number of queries.

Each of the following q lines contains one of the described queries in the following format:

  * \"1 pi\" (1 ≤ pi < [current width of sheet]) — the first type query. 
  * \"2 li ri\" (0 ≤ li < ri ≤ [current width of sheet]) — the second type query. 

Output

For each query of the second type, output the answer.

Examples

Input

7 4
1 3
1 2
2 0 1
2 1 2


Output

4
3


Input

10 9
2 2 9
1 1
2 0 1
1 8
2 0 8
1 2
2 1 3
1 4
2 2 4


Output

7
2
10
4
5

Note

The pictures below show the shapes of the paper during the queries of the first example:

<image>

After the first fold operation the sheet has width equal to 4, after the second one the width of the sheet equals to 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 101000;
int a[maxn], n, q;
inline int lowbit(int x) { return x & (-x); }
void add(int p, int v) {
  for (int i = p; i <= n; i += lowbit(i)) a[i] += v;
}
int sum(int p) {
  int sum = 0;
  for (int i = p; i; i -= lowbit(i)) sum += a[i];
  return sum;
}
int main() {
  scanf(\"%d%d\", &n, &q);
  int flag = 0;
  for (int i = 1; i <= n; i++) add(i, 1);
  int beg = 1, end = n;
  while (q--) {
    int k, p, l, r;
    scanf(\"%d\", &k);
    if (k == 1) {
      scanf(\"%d\", &p);
      if (p <= (end - beg + 1) / 2) {
        if (flag == 0) {
          for (int i = p + beg - 1, j = p + beg; i >= beg; i--, j++) {
            add(j, sum(i) - sum(i - 1));
          }
          beg = beg + p;
        } else {
          for (int i = end - p, j = end - p + 1; j <= end; i--, j++) {
            add(i, sum(j) - sum(j - 1));
          }
          end = end - p;
        }
      } else {
        if (flag == 0) {
          for (int i = beg + p - 1, j = p + beg; j <= end; j++, i--) {
            add(i, sum(j) - sum(j - 1));
          }
          end = beg + p - 1;
        } else {
          for (int i = end - p, j = end - p + 1; i >= beg; i--, j++) {
            add(j, sum(i) - sum(i - 1));
          }
          beg = end - p + 1;
        }
        flag = flag ^ 1;
      }
    } else if (k == 2) {
      scanf(\"%d%d\", &l, &r);
      l++;
      if (flag == 0) {
        l = beg + l - 1;
        r = beg + r - 1;
        printf(\"%d\n\", sum(r) - sum(l - 1));
      } else {
        l = end - l + 1;
        r = end - r + 1;
        printf(\"%d\n\", sum(l) - sum(r - 1));
      }
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re  # Fixed: Added missing import
import random
from bootcamp import Basebootcamp

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)

    def add(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def sum_query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

class Eapplemanandasheetofpaperbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_range = params.get('n_range', (5, 15))
        self.q_range = params.get('q_range', (3, 10))
        self.min_type2 = params.get('min_type2', 2)

    def case_generator(self):
        n = random.randint(*self.n_range)
        max_q = random.randint(*self.q_range)
        queries = []

        current_beg = 1
        current_end = n
        current_flag = 0
        type2_count = 0

        for _ in range(max_q):
            current_width = current_end - current_beg + 1
            if current_width <= 1 and type2_count >= self.min_type2:
                break  # Stop if paper too small and enough type2 queries

            # Prioritize generating type2 if needed
            if type2_count < self.min_type2:
                op_type = 2
            else:
                op_type = random.choices([1, 2], weights=[0.5, 0.5], k=1)[0]

            if op_type == 1:
                if current_width <= 1:
                    continue  # Cannot fold
                p = random.randint(1, current_width - 1)
                queries.append({'type': 1, 'p': p})

                # Update state based on fold logic
                if p <= (current_width) // 2:
                    if current_flag == 0:
                        current_beg += p
                    else:
                        current_end -= p
                else:
                    new_p = current_width - p
                    current_flag ^= 1
                    if current_flag == 0:
                        current_end = current_beg + new_p - 1
                    else:
                        current_beg = current_end - new_p + 1
            else:
                # Generate valid li and ri
                current_width = current_end - current_beg + 1
                if current_width == 0:
                    break  # No valid queries possible
                li = random.randint(0, current_width - 1)
                ri_max = current_width
                ri = random.randint(li + 1, ri_max)
                queries.append({'type': 2, 'li': li, 'ri': ri})
                type2_count += 1

        # Compute answers after generating all queries
        answers = self.compute_answers(n, queries)
        return {
            'n': n,
            'queries': queries,
            'answers': answers
        }

    def compute_answers(self, n, queries):
        ft = FenwickTree(n)
        for i in range(1, n+1):
            ft.add(i, 1)

        answers = []
        beg = 1
        end = n
        flag = 0

        for query in queries:
            if query['type'] == 1:
                p = query['p']
                current_width = end - beg + 1

                if p <= current_width // 2:
                    if flag == 0:
                        # Left folding (original direction)
                        for i in range(beg, beg + p):
                            val = ft.sum_query(i) - ft.sum_query(i - 1)
                            ft.add(beg + p + (i - beg), val)
                        beg += p
                    else:
                        # Left folding (reversed direction)
                        for j in range(end - p + 1, end + 1):
                            val = ft.sum_query(j) - ft.sum_query(j - 1)
                            ft.add(end - p - (j - (end - p + 1)), val)
                        end -= p
                else:
                    # Right folding (flip direction)
                    new_p = current_width - p
                    flag ^= 1
                    if flag == 0:
                        # Fold right in original direction
                        for j in range(beg + p, end + 1):
                            val = ft.sum_query(j) - ft.sum_query(j - 1)
                            ft.add(beg + p - 1 - (j - (beg + p)), val)
                        end = beg + p - 1
                    else:
                        # Fold right in reversed direction
                        for i in range(beg, beg + new_p):
                            val = ft.sum_query(i) - ft.sum_query(i - 1)
                            ft.add(end - new_p + 1 + (i - beg), val)
                        beg = end - new_p + 1
            else:
                li = query['li']
                ri = query['ri']
                current_width = end - beg + 1
                if flag == 0:
                    l = beg + li
                    r = beg + (ri - 1)
                    ans = ft.sum_query(r) - ft.sum_query(l - 1)
                else:
                    l = end - li
                    r = end - (ri - 1)
                    ans = ft.sum_query(l) - ft.sum_query(r - 1)
                answers.append(ans)
        return answers

    @staticmethod
    def prompt_func(question_case) -> str:
        problem_desc = (
            "Appleman has a very big sheet of paper with dimensions 1 × n. Your task is to help him with folding and counting operations.\n\n"
            "There are two types of queries:\n"
            "1. Fold the paper at position p. The left part (width p) is placed over the right part. The new width becomes (current width) - p.\n"
            "2. Calculate the total width between two cuts at positions l and r (0 ≤ l < r ≤ current width).\n\n"
            "Given the following queries, provide answers for each type 2 query in order. Place each answer within [answer] and [/answer] tags.\n\n"
            f"Initial width: {question_case['n']}\n"
            "Queries:\n"
        )
        for query in question_case['queries']:
            if query['type'] == 1:
                problem_desc += f"1. Fold at p={query['p']}\n"
            else:
                problem_desc += f"2. Query l={query['li']}, r={query['ri']}\n"
        problem_desc += "\nOutput each answer for type 2 queries in order, enclosed in [answer]...[/answer]."
        return problem_desc

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        if not answers:
            return None
        try:
            return [int(ans) for ans in answers]
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answers', [])
