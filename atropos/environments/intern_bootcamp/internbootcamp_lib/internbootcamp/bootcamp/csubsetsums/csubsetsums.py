"""# 

### 谜题描述
You are given an array a1, a2, ..., an and m sets S1, S2, ..., Sm of indices of elements of this array. Let's denote Sk = {Sk, i} (1 ≤ i ≤ |Sk|). In other words, Sk, i is some element from set Sk.

In this problem you have to answer q queries of the two types:

  1. Find the sum of elements with indices from set Sk: <image>. The query format is \"? k\". 
  2. Add number x to all elements at indices from set Sk: aSk, i is replaced by aSk, i + x for all i (1 ≤ i ≤ |Sk|). The query format is \"+ k x\". 



After each first type query print the required sum.

Input

The first line contains integers n, m, q (1 ≤ n, m, q ≤ 105). The second line contains n integers a1, a2, ..., an (|ai| ≤ 108) — elements of array a. 

Each of the following m lines describes one set of indices. The k-th line first contains a positive integer, representing the number of elements in set (|Sk|), then follow |Sk| distinct integers Sk, 1, Sk, 2, ..., Sk, |Sk| (1 ≤ Sk, i ≤ n) — elements of set Sk.

The next q lines contain queries. Each query looks like either \"? k\" or \"+ k x\" and sits on a single line. For all queries the following limits are held: 1 ≤ k ≤ m, |x| ≤ 108. The queries are given in order they need to be answered.

It is guaranteed that the sum of sizes of all sets Sk doesn't exceed 105.

Output

After each first type query print the required sum on a single line.

Please, do not write the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5 3 5
5 -5 5 1 -4
2 1 2
4 2 1 4 5
2 2 5
? 2
+ 3 4
? 1
+ 2 1
? 2


Output

-3
4
9

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
vector<int> adj[100009], bigs, smalls;
long long int a[100009], vals[100009], updates[100009];
int f[100009];
int store[320 + 10][100009], storeb[320 + 10][320 + 10];
int invb[100009], invs[100009];
int main() {
  int n, m, q;
  cin >> n >> m >> q;
  for (int i = 1; i <= n; i++) {
    cin >> a[i];
  }
  for (int i = 1; i <= m; i++) {
    int x;
    cin >> x;
    if (x > 320) {
      bigs.push_back(i);
      invb[i] = bigs.size() - 1;
    } else {
      smalls.push_back(i);
      invs[i] = smalls.size() - 1;
    }
    while (x--) {
      int y;
      cin >> y;
      adj[i].push_back(y);
      vals[i] += a[y];
    }
  }
  for (int i = 0; i < bigs.size(); i++) {
    int x = bigs[i];
    for (int j = 0; j < adj[x].size(); j++) {
      f[adj[x][j]]++;
    }
    for (int j = 0; j < smalls.size(); j++) {
      int y = smalls[j];
      for (int k = 0; k < adj[y].size(); k++) {
        if (f[adj[y][k]]) store[i][j]++;
      }
    }
    for (int j = 0; j < bigs.size(); j++) {
      int y = bigs[j];
      for (int k = 0; k < adj[y].size(); k++) {
        if (f[adj[y][k]]) storeb[i][j]++;
      }
    }
    for (int j = 0; j < adj[x].size(); j++) {
      f[adj[x][j]]--;
    }
  }
  while (q--) {
    string x;
    cin >> x;
    if (x[0] == '?') {
      int y;
      cin >> y;
      if (adj[y].size() > 320) {
        long long int sol = vals[y];
        for (int i = 0; i < bigs.size(); i++) {
          sol += updates[bigs[i]] * 1ll * storeb[invb[y]][i];
        }
        cout << sol << \"\n\";
      } else {
        long long int sol = 0;
        for (int j = 0; j < adj[y].size(); j++) {
          sol += a[adj[y][j]];
        }
        for (int j = 0; j < bigs.size(); j++) {
          sol += store[j][invs[y]] * 1ll * updates[bigs[j]];
        }
        cout << sol << \"\n\";
      }
    } else {
      int y, z;
      cin >> y >> z;
      if (adj[y].size() > 320) {
        updates[y] += z;
      } else {
        for (int j = 0; j < adj[y].size(); j++) {
          a[adj[y][j]] += z;
        }
        for (int j = 0; j < bigs.size(); j++) {
          vals[bigs[j]] += z * 1ll * store[j][invs[y]];
        }
      }
    }
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict

class Csubsetsumsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.threshold = params.get('threshold', 3)  # 使用较小的阈值方便测试

    def case_generator(self):
        threshold = self.threshold
        # Generate parameters with adjusted constraints
        n = random.randint(threshold + 1, 10)  # Ensure possible big sets
        m = random.randint(2, 5)
        q = random.randint(1, 5)

        # Generate initial array (1-based)
        array = [0] + [random.randint(-10, 10) for _ in range(n)]

        # Generate sets and classify big/small
        sets = []
        all_indices = list(range(1, n + 1))
        bigs = []
        smalls = []
        invb = dict()
        invs = dict()

        for k in range(1, m + 1):
            # Random size with higher probability to create big sets
            size = random.choice([2, threshold + 1])
            elements = random.sample(all_indices, size)
            sets.append(elements)
            if size > threshold:
                bigs.append(k)
                invb[k] = len(bigs) - 1
            else:
                smalls.append(k)
                invs[k] = len(smalls) - 1

        # Precompute store and storeb matrices
        store = []
        storeb = []
        for big_k in bigs:
            big_elements = set(sets[big_k - 1])
            # Compute store (interaction with small sets)
            s_counts = []
            for small_k in smalls:
                small_set = set(sets[small_k - 1])
                s_counts.append(len(big_elements & small_set))
            store.append(s_counts)
            # Compute storeb (interaction with big sets)
            b_counts = []
            for other_big in bigs:
                other_set = set(sets[other_big - 1])
                b_counts.append(len(big_elements & other_set))
            storeb.append(b_counts)

        # Initialize vals and updates
        vals = {k: sum(array[i] for i in sets[k-1]) for k in range(1, m+1)}
        updates = defaultdict(int)
        current_array = array.copy()  # Only modified by small sets

        # Generate queries
        queries = []
        valid_ks = list(range(1, m + 1))
        for _ in range(q):
            query_type = random.choice(['?', '+'])
            k = random.choice(valid_ks)
            if query_type == '+':
                x = random.randint(-10, 10)
                queries.append({'type': '+', 'k': k, 'x': x})
            else:
                queries.append({'type': '?', 'k': k})

        # Process queries and compute expected outputs
        expected_outputs = []
        for query in queries:
            if query['type'] == '+':
                k = query['k']
                x = query['x']
                if k in bigs:
                    updates[k] += x
                else:
                    # Update small set elements
                    indices = sets[k - 1]
                    for i in indices:
                        current_array[i] += x
                    # Update vals for all big sets
                    big_idx = invs[k]
                    for j, big_j in enumerate(bigs):
                        count = store[j][big_idx]
                        vals[big_j] += x * count
            else:
                k = query['k']
                if k in bigs:
                    big_idx = invb[k]
                    total = vals[k]
                    # Add contribution from other big sets
                    for j, big_j in enumerate(bigs):
                        total += updates[big_j] * storeb[big_idx][j]
                    expected_outputs.append(total)
                else:
                    small_idx = invs[k]
                    total = sum(current_array[i] for i in sets[k - 1])
                    # Add contribution from big sets
                    for j, big_j in enumerate(bigs):
                        total += updates[big_j] * store[j][small_idx]
                    expected_outputs.append(total)

        # Build case dictionary
        case = {
            'n': n,
            'm': m,
            'q': q,
            'array': array[1:],  # Save initial values (1-based)
            'sets': sets,
            'queries': queries,
            'expected_outputs': expected_outputs,
            '_bigs': bigs,
            '_smalls': smalls,
            '_store': store,
            '_storeb': storeb,
            '_updates': dict(updates),
            '_vals': vals
        }
        return case

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']} {question_case['q']}",
            ' '.join(map(str, question_case['array']))
        ]
        for s in question_case['sets']:
            input_lines.append(f"{len(s)} {' '.join(map(str, s))}")
        for q in question_case['queries']:
            if q['type'] == '?':
                input_lines.append(f"? {q['k']}")
            else:
                input_lines.append(f"+ {q['k']} {q['x']}")

        input_block = '\n'.join(input_lines)
        prompt = (
            "You are given an array and multiple sets of indices. Process a series of queries:\n"
            "1. '? k' - Sum elements of the k-th set\n"
            "2. '+ k x' - Add x to elements of the k-th set\n\n"
            "Input:\n" + input_block + "\n\n"
            "Output each sum result for '?', each on a new line within [answer] tags.\n"
            "Example:\n[answer]\n3\n-5\n[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        solutions = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    solutions.append(int(line))
                except ValueError:
                    return None
        return solutions

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('expected_outputs', [])
        return solution == expected
