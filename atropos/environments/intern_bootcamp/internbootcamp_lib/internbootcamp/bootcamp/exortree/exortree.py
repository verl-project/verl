"""# 

### 谜题描述
For a given sequence of distinct non-negative integers (b_1, b_2, ..., b_k) we determine if it is good in the following way:

  * Consider a graph on k nodes, with numbers from b_1 to b_k written on them.
  * For every i from 1 to k: find such j (1 ≤ j ≤ k, j≠ i), for which (b_i ⊕ b_j) is the smallest among all such j, where ⊕ denotes the operation of bitwise XOR (<https://en.wikipedia.org/wiki/Bitwise_operation#XOR>). Next, draw an undirected edge between vertices with numbers b_i and b_j in this graph.
  * We say that the sequence is good if and only if the resulting graph forms a tree (is connected and doesn't have any simple cycles). 



It is possible that for some numbers b_i and b_j, you will try to add the edge between them twice. Nevertheless, you will add this edge only once.

You can find an example below (the picture corresponding to the first test case). 

Sequence (0, 1, 5, 2, 6) is not good as we cannot reach 1 from 5.

However, sequence (0, 1, 5, 2) is good. 

<image>

You are given a sequence (a_1, a_2, ..., a_n) of distinct non-negative integers. You would like to remove some of the elements (possibly none) to make the remaining sequence good. What is the minimum possible number of removals required to achieve this goal?

It can be shown that for any sequence, we can remove some number of elements, leaving at least 2, so that the remaining sequence is good.

Input

The first line contains a single integer n (2 ≤ n ≤ 200,000) — length of the sequence.

The second line contains n distinct non-negative integers a_1, a_2, …, a_n (0 ≤ a_i ≤ 10^9) — the elements of the sequence.

Output

You should output exactly one integer — the minimum possible number of elements to remove in order to make the remaining sequence good.

Examples

Input


5
0 1 5 2 6


Output


1


Input


7
6 9 8 7 3 5 2


Output


2

Note

Note that numbers which you remove don't impact the procedure of telling whether the resulting sequence is good.

It is possible that for some numbers b_i and b_j, you will try to add the edge between them twice. Nevertheless, you will add this edge only once.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int best(int bit, vector<int> v) {
  if (bit == 0) {
    return v.size();
  }
  vector<int> set, unset;
  for (int i = 0; i < v.size(); i++) {
    if (((1 << bit) & v[i]) > 0)
      set.push_back(v[i]);
    else
      unset.push_back(v[i]);
  }
  int x = set.size();
  int y = unset.size();
  if (x < 2) {
    return x + best(bit - 1, unset);
  }
  if (y < 2) {
    return y + best(bit - 1, set);
  }
  if (x >= 2 && y >= 2) {
    return max(1 + best(bit - 1, set), 1 + best(bit - 1, unset));
  }
  return 0;
}
int main() {
  ios::sync_with_stdio(false), cin.tie(0), cout.tie(0);
  int n;
  cin >> n;
  vector<int> v;
  for (int i = 0; i < n; i++) {
    int x;
    cin >> x;
    v.push_back(x);
  }
  cout << n - best(30, v) << '\n';
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

def best(bit, v):
    if bit < 0:
        return 0
    if bit == 0:
        return len(v)
    set_bits = []
    unset_bits = []
    for num in v:
        if (num >> bit) & 1:
            set_bits.append(num)
        else:
            unset_bits.append(num)
    x = len(set_bits)
    y = len(unset_bits)
    if x < 2:
        return x + best(bit - 1, unset_bits)
    if y < 2:
        return y + best(bit - 1, set_bits)
    option1 = 1 + best(bit - 1, set_bits)
    option2 = 1 + best(bit - 1, unset_bits)
    return max(option1, option2)

class Exortreebootcamp(Basebootcamp):
    def __init__(self, n=5, max_val=10):
        self.n = n
        self.max_val = max_val
    
    def case_generator(self):
        a = random.sample(range(self.max_val), self.n)
        k = best(30, a)
        correct_removals = self.n - k
        return {
            'a': a,
            'correct_removals': correct_removals,
            'k': k
        }
    
    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        prompt = (
            f"给定一个由不同非负整数组成的序列：{a}。你需要删除尽可能少的元素，使得剩下的序列是'好的'。"
            f"一个序列是'好的'的条件是：按照以下方式构建的图是一个树结构。"
            f"对于每个元素b_i，找到与它进行XOR运算后结果最小的另一个元素b_j，并连接一条边。"
            f"图必须是连通且无环的。"
            f"请找出最小的删除数目，并将答案放在[answer]标签中，例如：[answer]3[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if matches:
            try:
                return int(matches[-1].strip())
            except ValueError:
                return None
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_removals = identity['correct_removals']
        return solution == correct_removals
