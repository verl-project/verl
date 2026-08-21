"""# 

### 谜题描述
You are given a sequence of positive integers x1, x2, ..., xn and two non-negative integers a and b. Your task is to transform a into b. To do that, you can perform the following moves:

  * subtract 1 from the current a; 
  * subtract a mod xi (1 ≤ i ≤ n) from the current a. 



Operation a mod xi means taking the remainder after division of number a by number xi.

Now you want to know the minimum number of moves needed to transform a into b.

Input

The first line contains a single integer n (1 ≤ n ≤ 105). The second line contains n space-separated integers x1, x2, ..., xn (2 ≤ xi ≤ 109). The third line contains two integers a and b (0 ≤ b ≤ a ≤ 109, a - b ≤ 106).

Output

Print a single integer — the required minimum number of moves needed to transform number a into number b.

Examples

Input

3
3 4 5
30 17


Output

6


Input

3
5 6 7
1000 200


Output

206

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a, b;
  scanf(\"%d\", &n);
  vector<int> v(n);
  for (int i = 0; i < n; i++) scanf(\"%d\", &v[i]);
  sort(v.begin(), v.end(), greater<int>());
  v.resize(unique(v.begin(), v.end()) - v.begin());
  scanf(\"%d%d\", &a, &b);
  int count = 0, be = 0;
  while (a > b && be < v.size()) {
    count++;
    int step = 0;
    for (int i = be; i < v.size() && v[i] > step; i++) {
      if (a - a % v[i] >= b)
        step = ((step > (a % v[i])) ? step : (a % v[i]));
      else
        be = i + 1;
    }
    if (step)
      a -= step;
    else
      a--;
  }
  count += a - b;
  printf(\"%d\n\", count);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from bootcamp import Basebootcamp

class Enumbertransformationiibootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=100, xi_min=2, xi_max=10**9, 
                 a_min=100, a_max=10**9, diff_min=1, diff_max=10**6):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.a_min = a_min
        self.a_max = a_max
        self.diff_min = diff_min
        self.diff_max = diff_max

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        xi = [random.randint(self.xi_min, self.xi_max) for _ in range(n)]
        xi = list(set(xi))  # 去重
        xi.sort(reverse=True)
        b = random.randint(0, self.a_max - self.diff_min)
        diff = random.randint(self.diff_min, self.diff_max)
        a = b + diff
        case = {
            'n': n,
            'xi': xi,
            'a': a,
            'b': b
        }
        return case

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        xi = question_case['xi']
        a = question_case['a']
        b = question_case['b']
        prompt = f"你有一个序列的正整数x1, x2, ..., xn，以及两个非负整数a和b。你的任务是将a转换为b，使用尽可能少的步骤。允许的操作有两种：1. 减去1；2. 减去a mod xi中的一个xi，其中xi是序列中的一个数。例如，如果a=30，xi=[3,4,5]，那么a mod 3是0，mod4是2，mod5是0。那么，可以选择减去0（这其实是无效操作，因为a不变），或者减去2。所以，操作后a变为28。请给定以下参数，求最小的操作次数：\n\nn = {n}\nxi = {xi}\na = {a}\nb = {b}\n\n请将你的答案放在[answer]标签中，例如：[answer]6[/answer]。"
        return prompt

    @staticmethod
    def extract_output(output):
        start_tag = "[answer]"
        end_tag = "[/answer]"
        start = output.rfind(start_tag)
        if start == -1:
            return None
        end = output.find(end_tag, start + len(start_tag))
        if end == -1:
            return None
        answer_str = output[start + len(start_tag):end].strip()
        if answer_str.isdigit():
            return int(answer_str)
        else:
            return None

    @staticmethod
    def compute_min_steps(a, b, xi):
        if a == b:
            return 0
        xi = sorted(list(set(xi)), reverse=True)
        count = 0
        be = 0
        while a > b and be < len(xi):
            max_step = 0
            new_be = be
            for i in range(be, len(xi)):
                mod = a % xi[i]
                if mod == 0:
                    continue
                if a - mod >= b:
                    if mod > max_step:
                        max_step = mod
                        new_be = i + 1
                else:
                    new_be = i + 1
                    break
            if max_step > 0:
                a -= max_step
                count += 1
                be = new_be
            else:
                a -= 1
                count += 1
        count += (a - b)
        return count

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        b = identity['b']
        xi = identity['xi']
        expected = cls.compute_min_steps(a, b, xi)
        return solution == expected
