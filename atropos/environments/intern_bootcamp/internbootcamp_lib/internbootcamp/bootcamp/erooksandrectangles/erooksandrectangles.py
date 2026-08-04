"""# 

### 谜题描述
Polycarpus has a chessboard of size n × m, where k rooks are placed. Polycarpus hasn't yet invented the rules of the game he will play. However, he has already allocated q rectangular areas of special strategic importance on the board, they must be protected well. According to Polycarpus, a rectangular area of ​​the board is well protected if all its vacant squares can be beaten by the rooks that stand on this area. The rooks on the rest of the board do not affect the area's defense. The position of the rooks is fixed and cannot be changed. We remind you that the the rook beats the squares located on the same vertical or horizontal line with it, if there are no other pieces between the square and the rook. Help Polycarpus determine whether all strategically important areas are protected.

Input

The first line contains four integers n, m, k and q (1 ≤ n, m ≤ 100 000, 1 ≤ k, q ≤ 200 000) — the sizes of the board, the number of rooks and the number of strategically important sites. We will consider that the cells of the board are numbered by integers from 1 to n horizontally and from 1 to m vertically. Next k lines contain pairs of integers \"x y\", describing the positions of the rooks (1 ≤ x ≤ n, 1 ≤ y ≤ m). It is guaranteed that all the rooks are in distinct squares. Next q lines describe the strategically important areas as groups of four integers \"x1 y1 x2 y2\" (1 ≤ x1 ≤ x2 ≤ n, 1 ≤ y1 ≤ y2 ≤ m). The corresponding rectangle area consists of cells (x, y), for which x1 ≤ x ≤ x2, y1 ≤ y ≤ y2. Strategically important areas can intersect of coincide.

Output

Print q lines. For each strategically important site print \"YES\" if it is well defended and \"NO\" otherwise.

Examples

Input

4 3 3 3
1 1
3 2
2 3
2 3 2 3
2 1 3 3
1 2 2 3


Output

YES
YES
NO

Note

Picture to the sample: <image> For the last area the answer is \"NO\", because cell (1, 2) cannot be hit by a rook.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
class ST {
 private:
  int st[400013];
  int size;
  int left(int w) { return w << 1; }
  int right(int w) { return (w << 1) + 1; }
  int queryI(int w, int L, int R, int a, int b) {
    if (a > R || b < L) return 2147483647;
    if (L >= a && R <= b) return st[w];
    int lC = queryI(left(w), L, (L + R) / 2, a, b);
    int rC = queryI(right(w), (L + R) / 2 + 1, R, a, b);
    return min(lC, rC);
  }
  void updateI(int w, int L, int R, int a, int val) {
    if (a > R || a < L) return;
    if (L == R)
      st[w] = val;
    else {
      updateI(left(w), L, (L + R) / 2, a, val);
      updateI(right(w), (L + R) / 2 + 1, R, a, val);
      st[w] = min(st[left(w)], st[right(w)]);
    }
  }

 public:
  ST(int s = 0) {
    fill(st, st + 4 * s, 0);
    size = s;
  }
  int query(int a, int b) { return queryI(1, 0, size - 1, a, b); }
  void update(int a, int val) { updateI(1, 0, size - 1, a, val); }
};
struct Query {
  int id, x1, x2, y1, y2;
};
int n, m, k, q;
vector<int> rooks[100013];
vector<int> rooks2[100013];
vector<Query> queries[100013];
vector<Query> queries2[100013];
int ans[200013];
ST tree;
int main() {
  int n, m, k, q;
  scanf(\"%d%d%d%d\", &n, &m, &k, &q);
  for (int i = 0; i < k; i++) {
    int x, y;
    scanf(\"%d%d\", &x, &y);
    rooks[x].push_back(y);
    rooks2[y].push_back(x);
  }
  for (int i = 0; i < q; i++) {
    int x1, x2, y1, y2;
    scanf(\"%d%d%d%d\", &x1, &y1, &x2, &y2);
    Query next;
    next.id = i;
    next.x1 = x1;
    next.x2 = x2;
    next.y1 = y1;
    next.y2 = y2;
    queries[x2].push_back(next);
    next.id = i;
    next.x1 = y1;
    next.x2 = y2;
    next.y1 = x1;
    next.y2 = x2;
    queries2[y2].push_back(next);
  }
  tree = ST(m + 1);
  for (int i = 1; i <= n; i++) {
    for (int j = 0; j < rooks[i].size(); j++) tree.update(rooks[i][j], i);
    for (int j = 0; j < queries[i].size(); j++) {
      Query cur = queries[i][j];
      if (tree.query(cur.y1, cur.y2) >= cur.x1) ans[cur.id] = 1;
    }
  }
  swap(rooks, rooks2);
  swap(queries, queries2);
  tree = ST(n + 1);
  for (int i = 1; i <= m; i++) {
    for (int j = 0; j < rooks[i].size(); j++) tree.update(rooks[i][j], i);
    for (int j = 0; j < queries[i].size(); j++) {
      Query cur = queries[i][j];
      if (tree.query(cur.y1, cur.y2) >= cur.x1) ans[cur.id] = 1;
    }
  }
  for (int i = 0; i < q; i++) printf(\"%s\n\", ans[i] ? \"YES\" : \"NO\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import bisect
from collections import defaultdict
from bootcamp import Basebootcamp

class Erooksandrectanglesbootcamp(Basebootcamp):
    def __init__(self, n_max=50, m_max=50, k_max=20, q_max=10):
        self.n_max = n_max
        self.m_max = m_max
        self.k_max = k_max
        self.q_max = q_max
    
    def case_generator(self):
        n = random.randint(1, self.n_max)
        m = random.randint(1, self.m_max)
        k = random.randint(1, min(self.k_max, n * m))
        
        positions = set()
        while len(positions) < k:
            x = random.randint(1, n)
            y = random.randint(1, m)
            positions.add((x, y))
        positions = list(positions)
        
        q = random.randint(1, self.q_max)
        queries = []
        for _ in range(q):
            x1 = random.randint(1, n)
            x2 = random.randint(x1, n)
            y1 = random.randint(1, m)
            y2 = random.randint(y1, m)
            queries.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
        
        return {
            'n': n,
            'm': m,
            'k': k,
            'q': q,
            'rooks': positions,
            'queries': queries
        }
    
    @staticmethod
    def prompt_func(question_case):
        case = question_case
        n, m, k, q = case['n'], case['m'], case['k'], case['q']
        input_lines = [f"{n} {m} {k} {q}"]
        input_lines.extend(f"{x} {y}" for x, y in case['rooks'])
        input_lines.extend(f"{q['x1']} {q['y1']} {q['x2']} {q['y2']}" for q in case['queries'])
        
        prompt = (
            "Determine if all cells in each strategic area are protected by rooks within the same area.\n"
            "Rules:\n"
            "1. A rook protects all cells in its row and column within the area, with no blocking.\n"
            "2. An area is protected if all empty cells are covered by at least one rook in the area.\n"
            "\n"
            "Input:\n" + '\n'.join(input_lines) +
            "\n\nOutput q lines of 'YES' or 'NO' wrapped in [answer]...[/answer] tags."
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        answer_block = matches[-1].strip()
        answers = []
        for line in answer_block.split('\n'):
            line = line.strip().upper()
            if line in ('YES', 'NO'):
                answers.append(line)
            elif line:
                return None
        return '\n'.join(answers) if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # Preprocess rook positions into column and row maps
        columns = defaultdict(list)
        rows = defaultdict(list)
        for x, y in identity['rooks']:
            bisect.insort(columns[y], x)
            bisect.insort(rows[x], y)
        
        correct = []
        for q in identity['queries']:
            x1, y1, x2, y2 = q['x1'], q['y1'], q['x2'], q['y2']
            valid = True
            
            # Check condition 1: All columns in [y1,y2] have a rook in [x1,x2]
            cond1 = True
            for y in range(y1, y2 + 1):
                if y not in columns:
                    cond1 = False
                    break
                # Find if any x in column y is within [x1,x2]
                idx = bisect.bisect_left(columns[y], x1)
                if idx < len(columns[y]) and columns[y][idx] <= x2:
                    continue
                else:
                    cond1 = False
                    break
            if cond1:
                correct.append("YES")
                continue
            
            # Check condition 2: All rows in [x1,x2] have a rook in [y1,y2]
            cond2 = True
            for x in range(x1, x2 + 1):
                if x not in rows:
                    cond2 = False
                    break
                # Find if any y in row x is within [y1,y2]
                idx = bisect.bisect_left(rows[x], y1)
                if idx < len(rows[x]) and rows[x][idx] <= y2:
                    continue
                else:
                    cond2 = False
                    break
            correct.append("YES" if cond2 else "NO")
        
        user_answers = solution.split('\n') if solution else []
        return len(user_answers) == len(correct) and all(u == c for u, c in zip(user_answers, correct))
