"""# 

### 谜题描述
Mashmokh's boss, Bimokh, didn't like Mashmokh. So he fired him. Mashmokh decided to go to university and participate in ACM instead of finding a new job. He wants to become a member of Bamokh's team. In order to join he was given some programming tasks and one week to solve them. Mashmokh is not a very experienced programmer. Actually he is not a programmer at all. So he wasn't able to solve them. That's why he asked you to help him with these tasks. One of these tasks is the following.

You have an array a of length 2n and m queries on it. The i-th query is described by an integer qi. In order to perform the i-th query you must:

  * split the array into 2n - qi parts, where each part is a subarray consisting of 2qi numbers; the j-th subarray (1 ≤ j ≤ 2n - qi) should contain the elements a[(j - 1)·2qi + 1], a[(j - 1)·2qi + 2], ..., a[(j - 1)·2qi + 2qi]; 
  * reverse each of the subarrays; 
  * join them into a single array in the same order (this array becomes new array a); 
  * output the number of inversions in the new a. 



Given initial array a and all the queries. Answer all the queries. Please, note that the changes from some query is saved for further queries.

Input

The first line of input contains a single integer n (0 ≤ n ≤ 20). 

The second line of input contains 2n space-separated integers a[1], a[2], ..., a[2n] (1 ≤ a[i] ≤ 109), the initial array.

The third line of input contains a single integer m (1 ≤ m ≤ 106). 

The fourth line of input contains m space-separated integers q1, q2, ..., qm (0 ≤ qi ≤ n), the queries.

Note: since the size of the input and output could be very large, don't use slow output techniques in your language. For example, do not use input and output streams (cin, cout) in C++.

Output

Output m lines. In the i-th line print the answer (the number of inversions) for the i-th query.

Examples

Input

2
2 1 4 3
4
1 2 0 2


Output

0
6
6
0


Input

1
1 2
3
0 1 1


Output

0
1
0

Note

If we reverse an array x[1], x[2], ..., x[n] it becomes new array y[1], y[2], ..., y[n], where y[i] = x[n - i + 1] for each i.

The number of inversions of an array x[1], x[2], ..., x[n] is the number of pairs of indices i, j such that: i < j and x[i] > x[j].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int xx[4] = {0, 0, 1, -1};
int yy[4] = {1, -1, 0, 0};
int n, m;
int a[int(1048576 + 2000)], b[int(1048576 + 2000)];
long long f[30][2];
long long res = 0;
void build(int l, int r, int h) {
  if (l == r) return;
  int mid = (l + r) / 2;
  build(l, mid, h + 1);
  build(mid + 1, r, h + 1);
  int i = l;
  int j = mid + 1;
  int k = l;
  while (i <= mid && j <= r) {
    if (a[i] <= a[j]) {
      b[k] = a[i];
      i++;
      f[h][1] += j - (mid + 1);
    } else {
      b[k] = a[j];
      j++;
      f[h][0] += (i - l);
    }
    k++;
  }
  while (i <= mid) {
    b[k] = a[i];
    i++;
    k++;
    f[h][1] += r - mid;
  }
  while (j <= r) {
    b[k] = a[j];
    j++;
    k++;
    f[h][0] += (mid - l + 1);
  }
  j = mid;
  int d = 0;
  for (int i = (l), _b = (mid); i <= _b; i++) {
    if (i == l || a[i] != a[i - 1]) d = 0;
    while (j + 1 <= r && a[j + 1] <= a[i]) {
      j++;
      if (a[j] == a[i]) d++;
    }
    f[h][0] -= d;
  }
  for (int i = (l), _b = (r); i <= _b; i++) a[i] = b[i];
}
void solve(int x) {
  for (int i = (x), _b = (n); i <= _b; i++) {
    res -= f[i][1];
    swap(f[i][1], f[i][0]);
    res += f[i][1];
  }
  printf(\"%I64d\n\", res);
}
int main() {
  scanf(\"%d\", &n);
  m = (1 << n);
  for (int i = (1), _b = (m); i <= _b; i++) scanf(\"%d\", &a[i]);
  build(1, m, 0);
  for (int i = (0), _b = (n); i <= _b; i++) res += f[i][1];
  int q;
  scanf(\"%d\", &q);
  for (int i = (1), _b = (q); i <= _b; i++) {
    int x;
    scanf(\"%d\", &x);
    solve(n - x);
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Emashmokhandreverseoperationbootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=5, min_val=1, max_val=100):
        self.max_n = max_n
        self.max_m = max_m
        self.min_val = min_val
        self.max_val = max_val
    
    def case_generator(self):
        n = random.randint(0, self.max_n)
        size = 2 ** n
        a = [random.randint(self.min_val, self.max_val) for _ in range(size)]
        m = random.randint(1, self.max_m)
        queries = [random.randint(0, n) for _ in range(m)]
        
        # Precompute inversion layers using reference solution approach
        current_a = a.copy()
        f = {i: [0, 0] for i in range(n+1)}  # Layer inversion counts
        
        def build(l, r, h):
            if l >= r: return
            mid = (l + r) // 2
            build(l, mid, h+1)
            build(mid+1, r, h+1)
            
            inv0 = 0
            inv1 = 0
            i, j = l, mid+1
            merged = []
            while i <= mid and j <= r:
                if current_a[i] <= current_a[j]:
                    merged.append(current_a[i])
                    inv1 += j - (mid+1)
                    i += 1
                else:
                    merged.append(current_a[j])
                    inv0 += i - l
                    j += 1
            
            while i <= mid:
                merged.append(current_a[i])
                inv1 += r - mid
                i += 1
                
            while j <= r:
                merged.append(current_a[j])
                inv0 += mid - l + 1
                j += 1
                
            for k in range(l, r+1):
                current_a[k] = merged[k-l]
            
            f[h][0] = inv0
            f[h][1] = inv1
        
        if n > 0:
            build(0, len(current_a)-1, 0)
        
        # Precompute initial total inversions
        total_inversions = sum(f[layer][1] for layer in f)
        
        answers = []
        for q in queries:
            # Apply query transformation by flipping layers
            layers_to_flip = [n - q] if q != 0 else []
            if q > 0:
                layers_to_flip += list(range(n - q + 1, n + 1))
            
            for layer in layers_to_flip:
                total_inversions -= f[layer][1]
                f[layer][0], f[layer][1] = f[layer][1], f[layer][0]
                total_inversions += f[layer][1]
            
            answers.append(total_inversions)
        
        return {
            'n': n,
            'a': a,
            'queries': queries,
            'answers': answers
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        a_str = ', '.join(map(str, question_case['a']))
        queries_str = ', '.join(map(str, question_case['queries']))
        m = len(question_case['queries'])
        prompt = f"""You are participating in an ACM programming contest. Your task is to process {m} queries on an array. Each query requires you to split the array into subarrays, reverse each, and compute the inversion count.

The initial array a has length 2^{question_case['n']} and is: [{a_str}].

There are {m} queries: {queries_str}.

For each query q_i (which is an integer value):
1. Split the array into 2^(n - q_i) subarrays, each of size 2^q_i, where n is {question_case['n']}
2. Reverse each subarray
3. Combine all reversed subarrays in the same order
4. Compute the number of inversions in the new array

Output your answers as {m} lines inside [answer] tags. Example:
[answer]
0
6
6
0
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        answers = []
        for line in last_block.splitlines():
            stripped = line.strip()
            if stripped:
                try:
                    answers.append(int(stripped))
                except ValueError:
                    pass
        return answers if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answers', [])

# For local testing
if __name__ == '__main__':
    sb = Emashmokhandreverseoperationbootcamp(max_n=3, max_m=3)
    case = sb.case_generator()
    print("Generated case:", case)
    print("Prompt example:")
    print(sb.prompt_func(case))
