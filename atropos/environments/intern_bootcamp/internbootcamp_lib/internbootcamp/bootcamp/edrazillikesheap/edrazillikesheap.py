"""# 

### 谜题描述
Drazil likes heap very much. So he created a problem with heap:

There is a max heap with a height h implemented on the array. The details of this heap are the following:

This heap contains exactly 2^h - 1 distinct positive non-zero integers. All integers are distinct. These numbers are stored in the array a indexed from 1 to 2^h-1. For any 1 < i < 2^h, a[i] < a[\left ⌊{i/2}\right ⌋].

Now we want to reduce the height of this heap such that the height becomes g with exactly 2^g-1 numbers in heap. To reduce the height, we should perform the following action 2^h-2^g times:

Choose an index i, which contains an element and call the following function f in index i:

<image>

Note that we suppose that if a[i]=0, then index i don't contain an element.

After all operations, the remaining 2^g-1 element must be located in indices from 1 to 2^g-1. Now Drazil wonders what's the minimum possible sum of the remaining 2^g-1 elements. Please find this sum and find a sequence of the function calls to achieve this value.

Input

The first line of the input contains an integer t (1 ≤ t ≤ 70 000): the number of test cases.

Each test case contain two lines. The first line contains two integers h and g (1 ≤ g < h ≤ 20). The second line contains n = 2^h-1 distinct positive integers a[1], a[2], …, a[n] (1 ≤ a[i] < 2^{20}). For all i from 2 to 2^h - 1, a[i] < a[\left ⌊{i/2}\right ⌋].

The total sum of n is less than 2^{20}.

Output

For each test case, print two lines.

The first line should contain one integer denoting the minimum sum after reducing the height of heap to g. The second line should contain 2^h - 2^g integers v_1, v_2, …, v_{2^h-2^g}. In i-th operation f(v_i) should be called.

Example

Input


2
3 2
7 6 3 5 4 2 1
3 2
7 6 5 4 3 2 1


Output


10
3 2 3 1
8
2 1 3 1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int a[(1 << 21) + 1];
int wid(int i) {
  if (a[2 * i] == 0 && a[2 * i + 1] == 0) {
    return i;
  } else {
    if (a[2 * i] > a[2 * i + 1]) {
      return wid(2 * i);
    } else {
      return wid(2 * i + 1);
    }
  }
}
void f(int i) {
  if (a[2 * i] == 0 && a[2 * i + 1] == 0)
    a[i] = 0;
  else {
    if (a[2 * i] > a[2 * i + 1]) {
      a[i] = a[2 * i];
      f(2 * i);
    } else {
      a[i] = a[2 * i + 1];
      f(2 * i + 1);
    }
  }
}
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  int h, g, t;
  cin >> t;
  while (t--) {
    cin >> h >> g;
    int n = (1 << h);
    vector<int> x;
    for (int i = 1; i < (1 << (h + 1)); ++i) {
      if (i < n)
        cin >> a[i];
      else
        a[i] = 0;
    }
    int index = (1 << g) - 1;
    for (int i = 1; i < (1 << g); ++i) {
      int t = wid(i);
      while (t > index) {
        x.push_back(i);
        f(i);
        if (x.size() == (1 << h) - (1 << g)) break;
        t = wid(i);
      }
      if (x.size() == (1 << h) - (1 << g)) break;
    }
    long long sum = 0;
    for (int i = 1; i < (1 << g); ++i) sum += a[i];
    cout << sum << '\n';
    for (int i = 0; i < x.size(); ++i) cout << x[i] << \" \";
    cout << '\n';
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from collections import deque
from bootcamp import Basebootcamp

class Edrazillikesheapbootcamp(Basebootcamp):
    def __init__(self, h=3, g=2):
        if g >= h or h < 1 or g < 1:
            raise ValueError("Invalid h and g values. Must satisfy 1 ≤ g < h.")
        self.h = h
        self.g = g

    def case_generator(self):
        a = self._generate_heap(self.h)
        correct_sum, correct_ops = self._compute_solution(self.h, self.g, a)
        return {
            'h': self.h,
            'g': self.g,
            'a': a,
            'correct_sum': correct_sum,
            'correct_operations': correct_ops,
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        h = question_case['h']
        g = question_case['g']
        a = question_case['a']
        prompt = f"""You are tasked with solving a heap height reduction problem. Given a max heap of height {h}, reduce it to height {g} by performing the minimal operations while achieving the smallest possible sum of the remaining elements.

**Problem Details:**
- The heap contains distinct positive integers arranged in a max-heap structure (each parent is larger than its children).
- Initial heap elements: {a}
- You must perform exactly {2**h - 2**g} operations of type f(i) as described below.

**Operation f(i):**
1. If node i has no children, set a[i] to 0.
2. Otherwise, replace a[i] with the larger of its two children and recursively apply f to that child node.

**Goal:**
- Minimize the sum of the remaining elements at indices 1 to {2**g - 1} after all operations.
- Provide the minimal sum and the sequence of function calls to achieve it.

**Input Format:**
- h = {h}, g = {g}
- Heap array: {a}

**Output Format:**
- First line: The minimal sum.
- Second line: Space-separated integers representing the sequence of function calls.

Place your final answer within [answer] and [/answer] tags. For example:
[answer]
10
3 2 3 1
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = last_match.split('\n')
        if len(lines) < 2:
            return None
        try:
            sum_val = int(lines[0].strip())
            ops = list(map(int, lines[1].strip().split()))
            return {'sum': sum_val, 'operations': ops}
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or 'sum' not in solution or 'operations' not in solution:
            return False
        sum_val = solution['sum']
        ops = solution['operations']
        correct_sum = identity['correct_sum']
        h, g = identity['h'], identity['g']
        expected_ops_len = (1 << h) - (1 << g)
        
        # Check length and sum match
        if len(ops) != expected_ops_len or sum_val != correct_sum:
            return False
        
        # Simulate operations
        try:
            a = [0] * (1 << (h + 1))
            original_a = identity['a']
            for i in range(len(original_a)):
                a[i+1] = original_a[i]
            for i in ops:
                if a[i] == 0:
                    return False
                cls._f_simulation(a, i)
            # Calculate final sum
            total = sum(a[i] for i in range(1, (1 << g)))
            return total == correct_sum
        except:
            return False

    @staticmethod
    def _f_simulation(a, i):
        current = i
        while True:
            left = 2 * current
            right = 2 * current + 1
            if a[left] == 0 and a[right] == 0:
                a[current] = 0
                break
            else:
                if a[left] > a[right]:
                    a[current] = a[left]
                    current = left
                else:
                    a[current] = a[right]
                    current = right

    def _generate_heap(self, h):
        size = (1 << h) - 1
        values = set()
        while len(values) < size:
            values.add(random.randint(1, (1 << 20) - 1))
        sorted_values = sorted(list(values), reverse=True)
        
        heap = [0] * size
        q = deque([0])
        idx = 0
        
        while q:
            pos = q.popleft()
            heap[pos] = sorted_values[idx]
            idx += 1
            left = 2 * pos + 1
            if left < size:
                q.append(left)
            right = 2 * pos + 2
            if right < size:
                q.append(right)
        return heap

    def _compute_solution(self, h, g, a_original):
        a = [0] * (1 << (h + 1))
        for i in range(len(a_original)):
            a[i+1] = a_original[i]
        target_ops = (1 << h) - (1 << g)
        x = []
        index_limit = (1 << g) - 1

        for i in range(1, (1 << g)):
            while len(x) < target_ops:
                t = self._wid(a, i)
                if t <= index_limit:
                    break
                x.append(i)
                self._f(a, i)
                if len(x) == target_ops: 
                    break
            if len(x) == target_ops:
                break

        sum_total = sum(a[i] for i in range(1, (1 << g)))
        return sum_total, x

    @staticmethod
    def _wid(a, i):
        current = i
        while True:
            left = 2 * current
            right = 2 * current + 1
            if a[left] == 0 and a[right] == 0:
                return current
            if a[left] > a[right]:
                current = left
            else:
                current = right

    @staticmethod
    def _f(a, i):
        current = i
        while True:
            left = 2 * current
            right = 2 * current + 1
            if a[left] == 0 and a[right] == 0:
                a[current] = 0
                break
            if a[left] > a[right]:
                a[current] = a[left]
                current = left
            else:
                a[current] = a[right]
                current = right
