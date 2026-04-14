"""# 

### 谜题描述
Rainbow built h cells in a row that are numbered from 1 to h from left to right. There are n cells with treasure. We call each of these n cells \"Treasure Cell\". The i-th \"Treasure Cell\" is the ai-th cell and the value of treasure in it is ci dollars.

Then, Freda went in the first cell. For now, she can go just k cells forward, or return to the first cell. That means Freda was able to reach the 1st, (k + 1)-th, (2·k + 1)-th, (3·k + 1)-th cells and so on.

Then Rainbow gave Freda m operations. Each operation is one of the following three types:

  1. Add another method x: she can also go just x cells forward at any moment. For example, initially she has only one method k. If at some moment she has methods a1, a2, ..., ar then she can reach all the cells with number in form <image>, where vi — some non-negative integer. 
  2. Reduce the value of the treasure in the x-th \"Treasure Cell\" by y dollars. In other words, to apply assignment cx = cx - y. 
  3. Ask the value of the most valuable treasure among the cells Freda can reach. If Freda cannot reach any cell with the treasure then consider the value of the most valuable treasure equal to 0, and do nothing. Otherwise take the most valuable treasure away. If several \"Treasure Cells\" have the most valuable treasure, take the \"Treasure Cell\" with the minimum number (not necessarily with the minimum number of cell). After that the total number of cells with a treasure is decreased by one. 



As a programmer, you are asked by Freda to write a program to answer each query.

Input

The first line of the input contains four integers: h (1 ≤ h ≤ 1018), n, m (1 ≤ n, m ≤ 105) and k (1 ≤ k ≤ 104).

Each of the next n lines contains two integers: ai (1 ≤ ai ≤ h), ci (1 ≤ ci ≤ 109). That means the i-th \"Treasure Cell\" is the ai-th cell and cost of the treasure in that cell is ci dollars. All the ai are distinct.

Each of the next m lines is in one of the three following formats:

  * \"1 x\" — an operation of type 1, 1 ≤ x ≤ h; 
  * \"2 x y\" — an operation of type 2, 1 ≤ x ≤ n, 0 ≤ y < cx; 
  * \"3\" — an operation of type 3. 



There are at most 20 operations of type 1. It's guaranteed that at any moment treasure in each cell has positive value. It's guaranteed that all operations is correct (no operation can decrease the value of the taken tresure).

Please, do not use the %lld specifier to read 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

For each operation of type 3, output an integer indicates the value (in dollars) of the most valuable treasure among the \"Treasure Cells\" Freda can reach. If there is no such treasure, output 0.

Examples

Input

10 3 5 2
5 50
7 60
8 100
2 2 5
3
1 3
3
3


Output

55
100
50

Note

In the sample, there are 10 cells and 3 \"Treasure Cells\". The first \"Treasure Cell\" is cell 5, having 50 dollars tresure in it. The second \"Treasure Cell\" is cell 7, having 60 dollars tresure in it. The third \"Treasure Cell\" is cell 8, having 100 dollars tresure in it.

At first, Freda can only reach cell 1, 3, 5, 7 and 9. In the first operation, we reduce the value in the second \"Treasure Cell\" from 60 to 55. Then the most valuable treasure among the \"Treasure Cells\" she can reach is max(50, 55) = 55. After the third operation, she can also go 3 cells forward each step, being able to reach cell 1, 3, 4, 5, 6, 7, 8, 9, 10. So the most valuable tresure is 100.

Noticed that she took the 55 dollars and 100 dollars treasure away, so the last answer is 50.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
struct pkt {
  long long x, c;
  bool jest;
};
const long long INF = 1000000000000000000;
void Dijkstra(long long k, vector<long long>& skoki,
              set<pair<long long, long long> >& S, vector<pkt>& a) {
  vector<long long> D(k, INF);
  D[0] = 0;
  set<pair<long long, long long> > Q;
  for (int i = 0; i < k; ++i) Q.insert(make_pair(D[i], i));
  while (Q.size() != 0) {
    pair<long long, long long> akt = *Q.begin();
    Q.erase(Q.begin());
    for (int i = 0; i < skoki.size(); ++i) {
      long long nowa_dl = akt.first + skoki[i];
      if (nowa_dl < D[nowa_dl % k]) {
        D[nowa_dl % k] = nowa_dl;
        Q.insert(make_pair(nowa_dl, nowa_dl % k));
      }
    }
  }
  for (int i = 0; i < a.size(); ++i) {
    if (!a[i].jest && D[a[i].x % k] <= a[i].x) {
      S.insert(make_pair(-a[i].c, i));
      a[i].jest = true;
    }
  }
}
int main() {
  ios_base::sync_with_stdio(0);
  long long h, n, m, k;
  cin >> h >> n >> m >> k;
  vector<pkt> a(n);
  for (int i = 0; i < n; ++i) {
    cin >> a[i].x >> a[i].c;
    --a[i].x;
    a[i].jest = false;
  }
  vector<long long> skoki;
  skoki.push_back(k);
  set<pair<long long, long long> > S;
  Dijkstra(k, skoki, S, a);
  for (int i = 0; i < m; ++i) {
    int op;
    cin >> op;
    if (op == 1) {
      long long x;
      cin >> x;
      skoki.push_back(x);
      Dijkstra(k, skoki, S, a);
    }
    if (op == 2) {
      long long x, y;
      cin >> x >> y;
      --x;
      if (S.find(make_pair(-a[x].c, x)) != S.end()) {
        S.erase(make_pair(-a[x].c, x));
        S.insert(make_pair(-a[x].c + y, x));
      }
      a[x].c -= y;
    }
    if (op == 3) {
      if (S.size() != 0) {
        pair<long long, long long> t = *S.begin();
        cout << -t.first << endl;
        S.erase(S.begin());
      } else
        cout << \"0\" << endl;
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import heapq
import random
import re
from bootcamp import Basebootcamp

class Cfetchthetreasurebootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=10, max_h=10**3, max_k=10**4, max_ops_type1=20, **params):
        self.max_n = max_n
        self.max_m = max_m
        self.max_h = max_h
        self.max_k = max_k
        self.max_ops_type1 = max_ops_type1

    def case_generator(self):
        h = random.randint(10, self.max_h)
        k = random.randint(1, min(self.max_k, h))
        n = random.randint(1, min(self.max_n, h))
        m = random.randint(1, self.max_m)

        # Generate distinct treasure positions
        ai_pool = random.sample(range(1, h + 1), n)
        treasures = [{'ai': ai, 'ci': random.randint(1, 10**9), 'taken': False} 
                    for ai in ai_pool]

        operations = []
        expected_outputs = []
        skoki = [k]
        type1_count = 0

        def compute_reachable():
            """Dijkstra算法计算可达余数的最小步数"""
            D = [float('inf')] * k
            D[0] = 0
            heap = [(0, 0)]
            visited = set()

            while heap:
                dist, r = heapq.heappop(heap)
                if r in visited:
                    continue
                visited.add(r)
                for s in skoki:
                    new_dist = dist + s
                    new_r = new_dist % k
                    if new_dist < D[new_r]:
                        D[new_r] = new_dist
                        heapq.heappush(heap, (new_dist, new_r))
            return D

        for _ in range(m):
            available_ops = []
            if type1_count < self.max_ops_type1:
                available_ops.append('1')
            available_ops.extend(['2', '3'])
            op_type = random.choice(available_ops) if available_ops else '3'

            if op_type == '1':
                x = random.randint(1, h)
                skoki.append(x)
                type1_count += 1
                operations.append(('1', x))
            
            elif op_type == '2':
                candidates = [i for i, t in enumerate(treasures) 
                            if not t['taken'] and t['ci'] > 1]
                if not candidates:
                    continue
                idx = random.choice(candidates)
                y = random.randint(1, treasures[idx]['ci'] - 1)
                treasures[idx]['ci'] -= y
                operations.append(('2', idx + 1, y))  # 1-based index
            
            else:  # type 3
                operations.append(('3',))
                D = compute_reachable()
                
                candidates = []
                for i, t in enumerate(treasures):
                    if t['taken']:
                        continue
                    
                    # 数学可达条件验证
                    remainder = (t['ai'] - 1) % k
                    if D[remainder] <= (t['ai'] - 1):
                        candidates.append((-t['ci'], t['ai'], i))  # 按ci降序，ai升序排序

                if not candidates:
                    expected_outputs.append(0)
                else:
                    candidates.sort()
                    selected_ci = -candidates[0][0]
                    expected_outputs.append(selected_ci)
                    treasures[candidates[0][2]]['taken'] = True

        return {
            'h': h,
            'n': n,
            'm': m,
            'k': k,
            'treasures': [{'ai': t['ai'], 'ci': t['ci']} for t in treasures],
            'operations': operations,
            'expected_outputs': expected_outputs
        }

    @staticmethod
    def prompt_func(question_case):
        treasures_str = '\n'.join(
            f"- Cell {t['ai']}: ${t['ci']}" 
            for t in question_case['treasures']
        )
        
        op_map = {
            '1': lambda x: f"Add step size {x}",
            '2': lambda x,y: f"Decrease treasure {x} by {y} (new value: {next(t['ci'] for t in question_case['treasures'] if t['ai'] == x)})",  # NoQA
            '3': lambda: "Query max treasure"
        }
        
        operations_str = []
        for op in question_case['operations']:
            if op[0] == '1':
                operations_str.append(f"Operation {len(operations_str)+1}: {op_map['1'](op[1])}")
            elif op[0] == '2':
                treasure_idx = op[1]-1
                original_ci = question_case['treasures'][treasure_idx]['ci'] + op[2]
                operations_str.append(f"Operation {len(operations_str)+1}: Decrease treasure at cell {question_case['treasures'][treasure_idx]['ai']} from {original_ci} to {original_ci - op[2]}")  # NoQA
            else:
                operations_str.append(f"Operation {len(operations_str)+1}: Query")

        prompt = f"""
        [Treasure Hunt Puzzle]
        Grid size: {question_case['h']} cells
        Available steps: Initially {question_case['k']} cells per jump
        
        [Treasure Cells (position: value)]
        {treasures_str}
        
        [Operations]
        {chr(10).join(operations_str)}
        
        [Rules]
        1. After each 'Add step' operation, new steps can be used immediately
        2. Treasure values remain positive after decrease
        3. Query operations select the maximum value among REACHABLE treasures
        4. Ties are broken by selecting the smallest position
        
        [Answer Format]
        Provide only the numerical results for each query operation in order, enclosed in [answer] tags:
        Example:
        [answer]
        55
        100
        50
        [/answer]
        """
        return prompt

    @staticmethod
    def extract_output(output):
        # 匹配所有可能的数值答案格式
        matches = re.findall(
            r'\[answer\][\s]*((?:-?\d+[\s]*)+)[\s]*\[/answer\]', 
            output, 
            re.IGNORECASE
        )
        
        if not matches:
            return None
            
        # 提取最后一个answer block中的数字
        numbers = []
        for line in matches[-1].strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # 提取行中所有可能数字
            found = re.findall(r'-?\d+', line)
            if found:
                numbers.extend(map(int, found))
        
        return numbers if numbers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_outputs']
