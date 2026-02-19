"""# 

### 谜题描述
It can be shown that any positive integer x can be uniquely represented as x = 1 + 2 + 4 + ... + 2k - 1 + r, where k and r are integers, k ≥ 0, 0 < r ≤ 2k. Let's call that representation prairie partition of x.

For example, the prairie partitions of 12, 17, 7 and 1 are: 

12 = 1 + 2 + 4 + 5,

17 = 1 + 2 + 4 + 8 + 2,

7 = 1 + 2 + 4,

1 = 1. 

Alice took a sequence of positive integers (possibly with repeating elements), replaced every element with the sequence of summands in its prairie partition, arranged the resulting numbers in non-decreasing order and gave them to Borys. Now Borys wonders how many elements Alice's original sequence could contain. Find all possible options!

Input

The first line contains a single integer n (1 ≤ n ≤ 105) — the number of numbers given from Alice to Borys.

The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 1012; a1 ≤ a2 ≤ ... ≤ an) — the numbers given from Alice to Borys.

Output

Output, in increasing order, all possible values of m such that there exists a sequence of positive integers of length m such that if you replace every element with the summands in its prairie partition and arrange the resulting numbers in non-decreasing order, you will get the sequence given in the input.

If there are no such values of m, output a single integer -1.

Examples

Input

8
1 1 2 2 3 4 5 8


Output

2 


Input

6
1 1 1 2 2 2


Output

2 3 


Input

5
1 2 4 4 4


Output

-1

Note

In the first example, Alice could get the input sequence from [6, 20] as the original sequence.

In the second example, Alice's original sequence could be either [4, 5] or [3, 3, 3].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 5;
long long int n, cnt1;
vector<long long int> v(N);
map<long long int, long long int> m;
bool f(long long int x) {
  m.clear();
  for (long long int i = (long long int)0; i <= (long long int)n - 1; i++)
    m[v[i]]++;
  vector<long long int> last(x, 1);
  long long int in = 2, cnt = x;
  m[1] -= x;
  while (m[in] > 0 && cnt > 0) {
    long long int inc = m[in];
    long long int cc = min(cnt, inc);
    m[in] -= cc;
    cnt = cc;
    for (long long int i = (long long int)0; i <= (long long int)cc - 1; i++)
      last[i] = in;
    in *= 2;
  }
  multiset<long long int> la;
  for (long long int i = (long long int)0; i <= (long long int)x - 1; i++)
    la.insert(last[i]);
  for (auto x : m) {
    long long int in = x.first, cnt = x.second;
    for (long long int i = (long long int)1; i <= (long long int)cnt; i++) {
      auto it = la.lower_bound((in + 1) / 2);
      if (it == la.end()) return false;
      la.erase(it);
    }
  }
  return true;
}
int main() {
  cin >> n;
  for (long long int i = (long long int)0; i <= (long long int)n - 1; i++)
    cin >> v[i], cnt1 += (v[i] == 1);
  long long int l = 0, r = cnt1 + 1;
  while (l < r - 1) {
    long long int m = l + (r - l) / 2;
    if (f(m))
      r = m;
    else
      l = m;
  }
  long long int mi = r;
  if (mi > cnt1) {
    cout << -1 << \"\n\";
    return 0;
  }
  for (long long int i = (long long int)mi; i <= (long long int)cnt1; i++)
    cout << i << \" \";
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
import bisect
from bootcamp import Basebootcamp

def compute_possible_m(a):
    cnt1 = a.count(1)
    n = len(a)
    a_sorted = sorted(a)
    freq = defaultdict(int)
    for num in a_sorted:
        freq[num] += 1

    def is_possible(m):
        current_freq = freq.copy()

        if current_freq.get(1, 0) < m:
            return False
        current_freq[1] -= m

        last = [1] * m
        current_power = 2
        cnt = m

        while current_freq.get(current_power, 0) > 0 and cnt > 0:
            available = current_freq[current_power]
            take = min(available, cnt)
            current_freq[current_power] -= take
            for i in range(take):
                last[i] = current_power
            cnt = take
            current_power *= 2

        last_sorted = sorted(last)
        remaining = []
        for num, count in sorted(current_freq.items()):
            if count > 0:
                remaining.extend([num] * count)

        for num in remaining:
            required = (num + 1) // 2
            idx = bisect.bisect_left(last_sorted, required)
            if idx >= len(last_sorted):
                return False
            del last_sorted[idx]
            bisect.insort(last_sorted, num)

        return True

    left, right = 0, cnt1 + 1
    while left < right - 1:
        mid = (left + right) // 2
        if is_possible(mid):
            right = mid
        else:
            left = mid
    mi = right

    if mi > cnt1:
        return [-1]
    return list(range(mi, cnt1 + 1))

class Eprairiepartitionbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.min_m = params.get('min_m', 1)
        self.max_m = params.get('max_m', 3)
        self.invalid_prob = params.get('invalid_prob', 0.3)

    def case_generator(self):
        if random.random() < self.invalid_prob:
            # Generate invalid case
            invalid_cases = [
                [1, 2, 4, 4, 4],
                [3, 3, 3],
                [5, 5],
                [2, 3, 3]
            ]
            a = random.choice(invalid_cases)
            a.sort()
        else:
            # Generate valid case
            m0 = random.randint(self.min_m, self.max_m)
            summands_list = []
            for _ in range(m0):
                k = random.randint(0, 3)
                sum_part = (2 ** k) - 1 if k > 0 else 0
                max_r = 2 ** k if k > 0 else 1
                r = random.randint(1, max_r)
                summands = [2 ** i for i in range(k)] + [r]
                summands_list.extend(summands)
            a = sorted(summands_list)
        
        possible_m = compute_possible_m(a)
        return {
            'n': len(a),
            'a': a,
            'correct_output': possible_m
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        prompt = f"""你是一个数学问题的解答者。根据以下描述解决该问题。

题目背景：

每个正整数x可以唯一表示为x = 1 + 2 + 4 + ... + 2^(k-1) + r的形式，其中k是非负整数，0 < r ≤ 2^k。这称为x的草原划分。例如，7的草原划分为1+2+4，12的草原划分为1+2+4+5。

Alice将原数列中的每个元素替换为其草原划分的所有加数，然后将所有加数按非降序排列得到一个序列。现在给定这个序列，需要找出所有可能的原数列的长度m的可能值。

输入格式：

第一行是一个整数n，表示序列的长度。
第二行有n个按非降序排列的整数a_1, a_2, ..., a_n。

输出格式：

输出所有可能的m的非降序排列，每个值之间用空格隔开。如果没有可能的m，输出-1。

现在，给定以下具体输入：

n = {n}
序列为：{a_str}

请仔细分析问题，确定所有可能的m的值。将你的答案放入[answer]和[/answer]的标签中。例如，若答案是2和3，则写成[answer]2 3[/answer]。"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            if last_match == '-1':
                return [-1]
            parts = last_match.split()
            solution = list(map(int, parts))
            if solution == [-1]:
                return solution
            if solution != sorted(solution):
                return None
            return solution
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        correct = identity['correct_output']
        return solution == correct
