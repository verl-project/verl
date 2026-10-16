"""# 

### 谜题描述
This is the harder version of the problem. In this version, 1 ≤ n, m ≤ 2⋅10^5. You can hack this problem if you locked it. But you can hack the previous problem only if you locked both problems.

You are given a sequence of integers a=[a_1,a_2,...,a_n] of length n. Its subsequence is obtained by removing zero or more elements from the sequence a (they do not necessarily go consecutively). For example, for the sequence a=[11,20,11,33,11,20,11]:

  * [11,20,11,33,11,20,11], [11,20,11,33,11,20], [11,11,11,11], [20], [33,20] are subsequences (these are just some of the long list); 
  * [40], [33,33], [33,20,20], [20,20,11,11] are not subsequences. 



Suppose that an additional non-negative integer k (1 ≤ k ≤ n) is given, then the subsequence is called optimal if:

  * it has a length of k and the sum of its elements is the maximum possible among all subsequences of length k; 
  * and among all subsequences of length k that satisfy the previous item, it is lexicographically minimal. 



Recall that the sequence b=[b_1, b_2, ..., b_k] is lexicographically smaller than the sequence c=[c_1, c_2, ..., c_k] if the first element (from the left) in which they differ less in the sequence b than in c. Formally: there exists t (1 ≤ t ≤ k) such that b_1=c_1, b_2=c_2, ..., b_{t-1}=c_{t-1} and at the same time b_t<c_t. For example:

  * [10, 20, 20] lexicographically less than [10, 21, 1], 
  * [7, 99, 99] is lexicographically less than [10, 21, 1], 
  * [10, 21, 0] is lexicographically less than [10, 21, 1]. 



You are given a sequence of a=[a_1,a_2,...,a_n] and m requests, each consisting of two numbers k_j and pos_j (1 ≤ k ≤ n, 1 ≤ pos_j ≤ k_j). For each query, print the value that is in the index pos_j of the optimal subsequence of the given sequence a for k=k_j.

For example, if n=4, a=[10,20,30,20], k_j=2, then the optimal subsequence is [20,30] — it is the minimum lexicographically among all subsequences of length 2 with the maximum total sum of items. Thus, the answer to the request k_j=2, pos_j=1 is the number 20, and the answer to the request k_j=2, pos_j=2 is the number 30.

Input

The first line contains an integer n (1 ≤ n ≤ 2⋅10^5) — the length of the sequence a.

The second line contains elements of the sequence a: integer numbers a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^9).

The third line contains an integer m (1 ≤ m ≤ 2⋅10^5) — the number of requests.

The following m lines contain pairs of integers k_j and pos_j (1 ≤ k ≤ n, 1 ≤ pos_j ≤ k_j) — the requests.

Output

Print m integers r_1, r_2, ..., r_m (1 ≤ r_j ≤ 10^9) one per line: answers to the requests in the order they appear in the input. The value of r_j should be equal to the value contained in the position pos_j of the optimal subsequence for k=k_j.

Examples

Input


3
10 20 10
6
1 1
2 1
2 2
3 1
3 2
3 3


Output


20
10
20
10
20
10


Input


7
1 2 1 3 1 2 1
9
2 1
2 2
3 1
3 2
3 3
1 1
7 1
7 7
7 4


Output


2
3
2
3
2
3
1
1
3

Note

In the first example, for a=[10,20,10] the optimal subsequences are: 

  * for k=1: [20], 
  * for k=2: [10,20], 
  * for k=3: [10,20,10]. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
vector<vector<int>> t;
int q(int v, int l, int r, int L, int R, int k) {
  if (l > R || r < L || L > R) return 0;
  if (l <= L && R <= r)
    return lower_bound(t[v].begin(), t[v].end(), k) - t[v].begin();
  return q(v * 2, l, r, L, (L + R) / 2, k) +
         q(v * 2 + 1, l, r, (L + R) / 2 + 1, R, k);
}
int32_t main() {
  int n, m, U = 1;
  cin.tie(0);
  ios::sync_with_stdio(false);
  cin >> n;
  vector<pair<int, int>> a(n);
  vector<int> u(n);
  for (int i = 0; i < n; i++)
    cin >> a[i].first, a[i].second = -i, u[i] = a[i].first;
  while (U < n) U <<= 1;
  n = U;
  while (a.size() < n) a.push_back({-1e9, -(int)a.size()}), u.push_back(1e9);
  t.resize(n * 2);
  vector<int> z(n + 1);
  sort(a.rbegin(), a.rend());
  for (int i = 0; i < n; i++) z[i] = -a[i].second, t[i + n].push_back(z[i]);
  for (int i = n - 1; i > 0; i--) {
    for (auto e : t[i * 2]) t[i].push_back(e);
    for (auto e : t[i * 2 + 1]) t[i].push_back(e);
    sort(t[i].begin(), t[i].end());
  }
  cin >> m;
  while (m--) {
    int a, b, l = 0, r = n + 1;
    cin >> a >> b;
    while (l < r - 1) {
      int m = (l + r) / 2;
      int ans = q(1, 0, a - 1, 0, n - 1, m);
      if (ans > b - 1)
        r = m;
      else
        l = m;
    }
    cout << u[l] << \"\n\";
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class D2optimalsubsequenceshardversionbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=5, a_min=1, a_max=100):
        self.max_n = max_n
        self.max_m = max_m
        self.a_min = a_min
        self.a_max = a_max
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        a = [random.randint(self.a_min, self.a_max) for _ in range(n)]
        m = random.randint(1, self.max_m)
        
        # 按照算法构造最优子序列
        sorted_with_index = sorted([(-val, idx) for idx, val in enumerate(a)])
        cache = {}
        
        for k in range(1, n+1):
            selected = sorted(sorted_with_index[:k], key=lambda x: x[1])
            cache[k] = [a[x[1]] for x in selected]
        
        queries = [(random.randint(1, n), random.randint(1, k)) for k in [random.randint(1, n) for _ in range(m)]]
        
        return {
            'n': n,
            'a': a,
            'm': m,
            'queries': queries,
            '_cache': cache
        }
    
    @staticmethod
    def prompt_func(question_case):
        a_str = ' '.join(map(str, question_case['a']))
        queries = '\n'.join(f"{k} {pos}" for k, pos in question_case['queries'])
        return f"""Given sequence: {a_str}
Answer {question_case['m']} queries about the optimal subsequence:
{queries}

Format answers in [answer] tags with one number per line:
[answer]
{{answers}}
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        answers = []
        for line in matches[-1].strip().split('\n'):
            line = line.strip()
            if re.fullmatch(r'\d+', line):
                answers.append(int(line))
        return answers
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != identity['m']:
            return False
        
        a = identity['a']
        queries = identity['queries']
        n = len(a)
        
        # 构建与参考代码相同的排序逻辑
        sorted_a = sorted([(-val, idx) for idx, val in enumerate(a)])
        
        for sol, (k, pos) in zip(solution, queries):
            if k > n or pos > k or pos < 1:
                return False
            
            # 精确重建选择过程
            selected = sorted(sorted_a[:k], key=lambda x: x[1])
            optimal_subseq = [a[x[1]] for x in selected]
            
            if pos-1 >= len(optimal_subseq) or sol != optimal_subseq[pos-1]:
                return False
        
        return True
