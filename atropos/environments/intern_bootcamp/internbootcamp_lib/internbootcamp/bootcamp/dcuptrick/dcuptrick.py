"""# 

### 谜题描述
The employees of the F company have lots of ways to entertain themselves. Today they invited a famous magician who shows a trick with plastic cups and a marble.

The point is to trick the spectator's attention. Initially, the spectator stands in front of a line of n plastic cups. Then the magician places a small marble under one cup and shuffles the cups. Then the spectator should guess which cup hides the marble.

But the head coder of the F company isn't easy to trick. When he saw the performance, he noticed several important facts:

  * each cup contains a mark — a number from 1 to n; all marks on the cups are distinct; 
  * the magician shuffles the cups in m operations, each operation looks like that: take a cup marked xi, sitting at position yi in the row of cups (the positions are numbered from left to right, starting from 1) and shift it to the very beginning of the cup row (on the first position). 



When the head coder came home after work he wanted to re-do the trick. Unfortunately, he didn't remember the starting or the final position of the cups. He only remembered which operations the magician performed. Help the coder: given the operations in the order they were made find at least one initial permutation of the cups that can go through the described operations in the given order. Otherwise, state that such permutation doesn't exist.

Input

The first line contains integers n and m (1 ≤ n, m ≤ 106). Each of the next m lines contains a couple of integers. The i-th line contains integers xi, yi (1 ≤ xi, yi ≤ n) — the description of the i-th operation of the magician. Note that the operations are given in the order in which the magician made them and the coder wants to make them in the same order.

Output

If the described permutation doesn't exist (the programmer remembered wrong operations), print -1. Otherwise, print n distinct integers, each from 1 to n: the i-th number should represent the mark on the cup that initially is in the row in position i.

If there are multiple correct answers, you should print the lexicographically minimum one.

Examples

Input

2 1
2 1


Output

2 1 


Input

3 2
1 2
1 1


Output

2 1 3 


Input

3 3
1 3
2 3
1 3


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
namespace treap {
struct Treap {
  Treap *l, *r;
  int pri, size, data;
  Treap(int d) : l(), r(), pri(std::rand()), size(1), data(d) {}
};
int size(Treap* t) { return t ? t->size : 0; }
void pull(Treap* t) { t->size = size(t->l) + size(t->r) + 1; }
Treap* merge(Treap* a, Treap* b) {
  if (!a) return b;
  if (!b) return a;
  if (a->pri > b->pri) {
    a->r = merge(a->r, b);
    pull(a);
    return a;
  } else {
    b->l = merge(a, b->l);
    pull(b);
    return b;
  }
}
void split(Treap* t, int k, Treap*& a, Treap*& b) {
  if (!t)
    a = b = 0;
  else if (size(t->l) <= k) {
    a = t;
    split(t->r, k - size(t->l) - 1, a->r, b);
    pull(a);
  } else {
    b = t;
    split(t->l, k, a, b->l);
    pull(b);
  }
}
Treap* remove(Treap* t, int k) {
  Treap *a, *b, *c, *d;
  split(t, k - 1, a, b);
  split(b, k - size(a), c, d);
  return merge(a, d);
}
Treap* kth(Treap* t, int k) {
  if (k < size(t->l))
    return kth(t->l, k);
  else if (k == size(t->l))
    return t;
  else
    return kth(t->r, k - size(t->l) - 1);
}
}  // namespace treap
const int maxn = 1000000;
int val[maxn];
bool used[maxn];
using namespace treap;
int main() {
  std::srand(std::time(0));
  int n, m;
  std::scanf(\"%d%d\", &n, &m);
  Treap* treap = new Treap(0);
  for (int i = 1; i < n; i++) treap = merge(treap, new Treap(i));
  std::fill_n(val, n, -1);
  for (int i = 0; i < m; i++) {
    int x, y;
    std::scanf(\"%d%d\", &x, &y);
    x--;
    y--;
    int opos = kth(treap, y)->data;
    int fndval = val[opos];
    if (fndval != -1 && fndval != x) {
      std::puts(\"-1\");
      return 0;
    }
    if (fndval == -1) {
      if (used[x]) {
        std::puts(\"-1\");
        return 0;
      } else {
        used[x] = true;
        val[opos] = x;
      }
    }
    treap = remove(treap, y);
    treap = merge(new Treap(opos), treap);
  }
  for (int i = 0, j = 0; i < n; i++) {
    int fnd = val[i];
    if (fnd == -1) {
      while (used[j]) j++;
      used[j] = true;
      fnd = j;
    }
    std::printf(\"%d%c\", fnd + 1, i == n - 1 ? '\n' : ' ');
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from typing import List, Tuple, Union

class Dcuptrickbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', random.randint(2, 5))
        self.m = params.get('m', random.randint(1, 3))
        self.valid_probability = params.get('valid_probability', 0.5)
    
    def case_generator(self):
        n = self.n
        m = self.m
        operations = []
        for _ in range(m):
            x = random.randint(1, n)
            y = random.randint(1, n)
            operations.append((x, y))
        
        expected_solution = self.solve(n, m, operations)
        return {
            'n': n,
            'm': m,
            'operations': operations,
            'expected_solution': expected_solution
        }
    
    @staticmethod
    def solve(n: int, m: int, operations: List[Tuple[int, int]]) -> Union[List[int], int]:
        val = [-1] * n
        used = [False] * n
        current = list(range(n))
        valid = True

        for op in operations:
            x, y = op
            x -= 1  # 0-based
            y -= 1

            if y >= len(current):
                valid = False
                break
            opos = current[y]

            if val[opos] != -1:
                if val[opos] != x:
                    valid = False
                    break
            else:
                if used[x]:
                    valid = False
                    break
                val[opos] = x
                used[x] = True

            # Move to front
            current.pop(y)
            current.insert(0, opos)

        if not valid:
            return -1

        j = 0
        for i in range(n):
            if val[i] == -1:
                while j < n and used[j]:
                    j += 1
                if j >= n:
                    return -1
                val[i] = j
                used[j] = True

        solution = [v + 1 for v in val]
        return solution
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        operations = question_case['operations']
        input_example = f"{n} {m}\n" + "\n".join(f"{x} {y}" for x, y in operations)
        problem = (
            "The employees of the F company witnessed a magician's trick with cups and a marble. "
            "Your task is to determine the initial permutation of cups or state it's impossible.\n\n"
            
            "Rules:\n"
            "1. Cups are numbered 1 to n. Each operation moves the cup at position yi to the front.\n"
            "2. Given m operations in order, find the lexicographically smallest initial permutation.\n"
            "3. If impossible, output -1.\n\n"
            
            "Input:\n"
            f"{input_example}\n\n"
            
            "Output:\n"
            "The lexicographically smallest initial permutation or -1.\n\n"
            
            "Format your answer within [answer] and [/answer]. Example: [answer]2 1 3[/answer] or [answer]-1[/answer]."
        )
        return problem
    
    @staticmethod
    def extract_output(output: str) -> Union[List[int], int, None]:
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if last_match == '-1':
            return -1
        try:
            solution = list(map(int, last_match.split()))
            if all(1 <= num <= 1000000 for num in solution):
                return solution
            else:
                return None
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity) -> bool:
        expected = identity['expected_solution']
        if expected == -1:
            return solution == -1
        else:
            return solution == expected
