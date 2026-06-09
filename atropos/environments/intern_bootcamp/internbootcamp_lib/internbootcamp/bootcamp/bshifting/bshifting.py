"""# 

### 谜题描述
John Doe has found the beautiful permutation formula.

Let's take permutation p = p1, p2, ..., pn. Let's define transformation f of this permutation: 

<image>

where k (k > 1) is an integer, the transformation parameter, r is such maximum integer that rk ≤ n. If rk = n, then elements prk + 1, prk + 2 and so on are omitted. In other words, the described transformation of permutation p cyclically shifts to the left each consecutive block of length k and the last block with the length equal to the remainder after dividing n by k. 

John Doe thinks that permutation f(f( ... f(p = [1, 2, ..., n], 2) ... , n - 1), n) is beautiful. Unfortunately, he cannot quickly find the beautiful permutation he's interested in. That's why he asked you to help him.

Your task is to find a beautiful permutation for the given n. For clarifications, see the notes to the third sample.

Input

A single line contains integer n (2 ≤ n ≤ 106).

Output

Print n distinct space-separated integers from 1 to n — a beautiful permutation of size n.

Examples

Input

2


Output

2 1 


Input

3


Output

1 3 2 


Input

4


Output

4 2 3 1 

Note

A note to the third test sample: 

  * f([1, 2, 3, 4], 2) = [2, 1, 4, 3]
  * f([2, 1, 4, 3], 3) = [1, 4, 2, 3]
  * f([1, 4, 2, 3], 4) = [4, 2, 3, 1]

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int di[] = {-1, 0, 1, 0};
const int dj[] = {0, 1, 0, -1};
const long long MOD = 1e9 + 7;
const long long INF = 1e9;
const double EPS = 1e-5;
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  int n;
  cin >> n;
  deque<int> num(n);
  for (int i = 0; i < n; i++) num[i] = i + 1;
  for (int shft = 2; shft <= n; shft++) {
    int last = num[0];
    for (int i = shft; i < n; i += shft) swap(num[i], last);
    num.pop_front();
    num.push_back(last);
  }
  for (int i = 0; i < n; i++) cout << num[i] << \" \n\"[i == n - 1];
  cin.ignore(), cin.get();
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque
from bootcamp import Basebootcamp

class Bshiftingbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        参数有效性检查：确保2 ≤ min_n ≤ max_n ≤ 1e6
        """
        self.min_n = max(2, params.get('min_n', 2))
        self.max_n = min(10**6, params.get('max_n', 100))
        super().__init__(**params)

    def case_generator(self):
        """
        严格保证生成的n在[2, 1e6]范围内
        """
        n = random.randint(self.min_n, self.max_n)
        return {'n': n}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""Find the beautiful permutation for n={n} by applying sequential transformations for k=2 to n.
Each transformation divides the permutation into k-length blocks and left-rotates each block.
Output format: space-separated numbers within [answer] tags.

Example for n=4:
[answer]
4 2 3 1
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强提取逻辑：允许任意分隔符和空格
        matches = re.findall(r'\[answer](.*?)\[/answer]', output, re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
            
        # 提取最后一个answer块并规范化格式
        raw_answer = matches[-1].strip()
        processed = re.sub(r'[^\d]', ' ', raw_answer)  # 非数字转空格
        processed = re.sub(r'\s+', ' ', processed).strip()  # 合并空格
        return processed

    @classmethod
    def _generate_correct_answer(cls, n):
        """严格遵循C++参考实现逻辑"""
        num = deque(range(1, n+1))
        for k in range(2, n+1):
            if len(num) == 0:
                break
            last = num[0]
            i = k
            while i < n:  # 注意这里是原始n值，不是当前deque长度
                if i < len(num):
                    num[i], last = last, num[i]
                i += k
            num.popleft()
            num.append(last)
        return list(num)

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        try:
            # 严格验证数字集合和顺序
            user_ans = list(map(int, solution.split()))
            correct = cls._generate_correct_answer(n)
            return user_ans == correct
        except:
            return False
