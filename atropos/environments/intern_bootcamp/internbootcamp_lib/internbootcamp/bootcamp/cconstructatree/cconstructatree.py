"""# 

### 谜题描述
Misha walked through the snowy forest and he was so fascinated by the trees to decide to draw his own tree!

Misha would like to construct a rooted tree with n vertices, indexed from 1 to n, where the root has index 1. Every other vertex has a parent p_i, and i is called a child of vertex p_i. Vertex u belongs to the subtree of vertex v iff v is reachable from u while iterating over the parents (u, p_{u}, p_{p_{u}}, ...). Clearly, v belongs to its own subtree, and the number of vertices in the subtree is called the size of the subtree. Misha is only interested in trees where every vertex belongs to the subtree of vertex 1.

Below there is a tree with 6 vertices. The subtree of vertex 2 contains vertices 2, 3, 4, 5. Hence the size of its subtree is 4. 

<image>

The branching coefficient of the tree is defined as the maximum number of children in any vertex. For example, for the tree above the branching coefficient equals 2. Your task is to construct a tree with n vertices such that the sum of the subtree sizes for all vertices equals s, and the branching coefficient is minimum possible.

Input

The only input line contains two integers n and s — the number of vertices in the tree and the desired sum of the subtree sizes (2 ≤ n ≤ 10^5; 1 ≤ s ≤ 10^{10}).

Output

If the required tree does not exist, output «No». Otherwise output «Yes» on the first line, and in the next one output integers p_2, p_3, ..., p_n, where p_i denotes the parent of vertex i.

Examples

Input


3 5


Output


Yes
1 1 


Input


4 42


Output


No


Input


6 15


Output


Yes
1 2 3 1 5 

Note

Below one can find one of the possible solutions for the first sample case. The sum of subtree sizes equals 3 + 1 + 1 = 5, and the branching coefficient equals 2.

<image>

Below one can find one of the possible solutions for the third sample case. The sum of subtree sizes equals 6 + 3 + 2 + 1 + 2 + 1 = 15, and the branching coefficient equals 2.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long s;
int n;
const int maxN = (int)1e5 + 10;
int p[maxN];
int d[maxN];
bool go(int deg) {
  int he = 2;
  long long curs = s;
  long long can = deg;
  long long already = 0;
  d[1] = 1;
  curs--;
  for (int i = 2; i <= n; i++) {
    if (already == can) {
      he++;
      already = 0;
      can *= deg;
    }
    long long mx = ((1LL * he + 1 + he + 1 - i - 1 + n) * (n - i)) / 2;
    if (curs <= he + mx) {
      already++;
      d[i] = he;
      curs -= he;
    } else {
      he++;
      d[i] = he;
      curs -= he;
    }
  }
  return (curs == 0);
}
int can[maxN];
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  cin >> n >> s;
  if ((2 * s > 1LL * n * (n + 1)) || (s < 2 * n - 1)) {
    cout << \"No\";
    return 0;
  }
  int l = 0;
  int r = n - 1;
  while (r - l > 1) {
    int mid = (l + r) / 2;
    if (go(mid)) {
      r = mid;
    } else {
      l = mid;
    }
  }
  go(r);
  cout << \"Yes\" << '\n';
  int le = 1;
  for (int i = 1; i <= n; i++) {
    can[i] = r;
  }
  for (int i = 2; i <= n; i++) {
    if (can[le] == 0) le++;
    while (d[le] + 1 < d[i]) le++;
    p[i] = le;
    can[le]--;
    cout << p[i] << \" \";
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
from bootcamp import Basebootcamp
import re

class Cconstructatreebootcamp(Basebootcamp):
    def __init__(self, max_n=1000, **params):
        self.max_n = max_n
        self.params = params
    
    def case_generator(self):
        n = random.randint(2, self.max_n)
        generate_possible = random.choice([True, False])
        
        s_min = 2 * n - 1
        s_max = n * (n + 1) // 2
        
        if generate_possible:
            s = random.randint(s_min, s_max)
        else:
            if random.random() < 0.5:
                s = random.randint(1, s_min - 1)
            else:
                s = s_max + random.randint(1, 1000)
        
        if s < s_min or s > s_max:
            return {'n': n, 's': s, 'possible': False}
        else:
            solution = self.generate_solution(n, s)
            if solution['possible']:
                return {
                    'n': n,
                    's': s,
                    'possible': True,
                    'p_array': solution['p_array'],
                    'k': solution['k']
                }
            else:
                return {'n': n, 's': s, 'possible': False}
    
    @staticmethod
    def generate_solution(n, s):
        if s < 2 * n -1 or s > n * (n + 1) // 2:
            return {'possible': False}
        
        left = 0
        right = n - 1
        d_final = None
        answer_r = None
        
        while right - left > 1:
            mid = (left + right) // 2
            possible, d = Cconstructatreebootcamp.go(mid, n, s)
            if possible:
                right = mid
            else:
                left = mid
        
        possible, d = Cconstructatreebootcamp.go(right, n, s)
        if not possible:
            possible_left, d_left = Cconstructatreebootcamp.go(left, n, s)
            if possible_left:
                right = left
                d = d_left
            else:
                return {'possible': False}
        
        p_array = Cconstructatreebootcamp.construct_p(n, right, d)
        children = defaultdict(list)
        for i in range(2, n + 1):
            parent = p_array[i-2]
            children[parent].append(i)
        max_degree = max(len(v) for v in children.values()) if children else 0
        
        return {
            'possible': True,
            'p_array': p_array,
            'k': right,
            'max_degree': max_degree
        }
    
    @staticmethod
    def go(deg, n, s):
        he = 2
        curs = s
        curs -= 1  # Root node's contribution
        already = 0
        can = deg
        d = [0] * (n + 1)
        d[1] = 1  # Depth of root is 1
        
        for i in range(2, n + 1):
            if already == can:
                he += 1
                already = 0
                can *= deg
            
            remaining_nodes = n - i
            mx_term = (2 * he + remaining_nodes) * (remaining_nodes) // 2
            
            if curs <= he + mx_term:
                already += 1
                d[i] = he
                curs -= he
            else:
                he += 1
                d[i] = he
                curs -= he
        
        return curs == 0, d
    
    @staticmethod
    def construct_p(n, r, d):
        can = [r] * (n + 2)
        le = 1
        p = [0] * (n + 1)
        
        for i in range(2, n + 1):
            while le <= n and can[le] == 0:
                le += 1
            
            while le < i and d[le] + 1 < d[i]:
                le += 1
            
            p[i] = le
            can[le] -= 1
        
        return p[2:n+1]
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        s = question_case['s']
        problem_desc = f"""Misha wants to construct a rooted tree with {n} vertices where the root is vertex 1. The tree must satisfy that the sum of the sizes of all subtrees equals {s}. The branching coefficient (maximum number of children any vertex has) must be as small as possible. 

Your task is to determine if such a tree exists. If it exists, output "Yes" followed by the parent array. Otherwise, output "No". 

The parent array should list the parent of vertices 2 to {n}, space-separated. For example, if the parents are 1 and 1 for vertices 2 and 3 respectively, output "1 1". 

Please format your answer as follows:

[answer]
Yes
1 1 
[/answer]

or 

[answer]
No
[/answer]

Ensure your answer is enclosed within [answer] and [/answer] tags."""
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_block = matches[-1].strip()
        lines = [line.strip() for line in answer_block.split('\n') if line.strip()]
        if not lines:
            return None
        first_line = lines[0].lower()
        if first_line == 'no':
            return 'No'
        elif first_line == 'yes' and len(lines) >= 2:
            return 'Yes\n' + ' '.join(lines[1].split())
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not identity.get('possible', False):
            return solution.strip().lower() == 'no'
        else:
            lines = solution.strip().split('\n')
            if len(lines) < 1 or lines[0].lower() != 'yes':
                return False
            if len(lines) < 2:
                return False
            p_str = lines[1].strip()
            try:
                p = list(map(int, p_str.split()))
            except:
                return False
            n = identity['n']
            if len(p) != n - 1:
                return False
            children = defaultdict(list)
            valid = True
            for i in range(2, n + 1):
                parent = p[i-2]
                if parent < 1 or parent >= i:
                    valid = False
                children[parent].append(i)
            if not valid:
                return False
            subtree_sizes = [1] * (n + 1)
            for node in range(n, 0, -1):
                for child in children.get(node, []):
                    subtree_sizes[node] += subtree_sizes[child]
            total_sum = sum(subtree_sizes)
            if total_sum != identity['s']:
                return False
            max_degree = max((len(v) for v in children.values()), default=0)
            if max_degree != identity.get('k', 0):
                return False
            return True
