"""# 

### 谜题描述
The weight of a sequence is defined as the number of unordered pairs of indexes (i,j) (here i < j) with same value (a_{i} = a_{j}). For example, the weight of sequence a = [1, 1, 2, 2, 1] is 4. The set of unordered pairs of indexes with same value are (1, 2), (1, 5), (2, 5), and (3, 4).

You are given a sequence a of n integers. Print the sum of the weight of all subsegments of a. 

A sequence b is a subsegment of a sequence a if b can be obtained from a by deletion of several (possibly, zero or all) elements from the beginning and several (possibly, zero or all) elements from the end.

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 10^5). Description of the test cases follows.

The first line of each test case contains a single integer n (1 ≤ n ≤ 10^5).

The second line of each test case contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 10^9).

It is guaranteed that the sum of n over all test cases does not exceed 10^5.

Output

For each test case, print a single integer — the sum of the weight of all subsegments of a.

Example

Input


2
4
1 2 1 1
4
1 2 3 4


Output


6
0

Note

  * In test case 1, all possible subsegments of sequence [1, 2, 1, 1] having size more than 1 are: 
    1. [1, 2] having 0 valid unordered pairs; 
    2. [2, 1] having 0 valid unordered pairs; 
    3. [1, 1] having 1 valid unordered pair; 
    4. [1, 2, 1] having 1 valid unordered pairs; 
    5. [2, 1, 1] having 1 valid unordered pair; 
    6. [1, 2, 1, 1] having 3 valid unordered pairs. 
Answer is 6.
  * In test case 2, all elements of the sequence are distinct. So, there is no valid unordered pair with the same value for any subarray. Answer is 0. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from itertools import repeat
from collections import defaultdict
def solve():
    n = int(stdin.readline())
    a = map(int, stdin.readline().split(), repeat(10, n))
    s = defaultdict(int)
    ans = 0
    for i, x in enumerate(a):
        ans += s[x] * (n - i)
        s[x] += i + 1
    print ans

T = int(stdin.readline())
for t in xrange(T):
    solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict
from random import randint
import re
from bootcamp import Basebootcamp

class Csequencepairweightbootcamp(Basebootcamp):
    def __init__(self, max_t=2, max_n=5, a_max=5):
        self.max_t = max_t
        self.max_n = max_n
        self.a_max = a_max  # 控制元素的范围
    
    def case_generator(self):
        t = randint(1, self.max_t)
        cases = []
        for _ in range(t):
            n = randint(1, self.max_n)
            a = [randint(1, self.a_max) for _ in range(n)]
            output = self._calculate_single_case(n, a)
            cases.append({
                'n': n,
                'a': a,
                'output': output
            })
        identity = {
            't': t,
            'cases': cases
        }
        return identity
    
    @staticmethod
    def _calculate_single_case(n, a):
        s = defaultdict(int)
        ans = 0
        for i, x in enumerate(a):
            ans += s[x] * (n - i)
            s[x] += (i + 1)
        return ans
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['t'])]
        for case in question_case['cases']:
            input_lines.append(str(case['n']))
            input_lines.append(' '.join(map(str, case['a'])))
        input_str = '\n'.join(input_lines)
        prompt = f"""你是编程竞赛的参赛者，请解决以下问题：

问题描述：

给定多个测试用例，每个测试用例要求计算数组所有子段的权重总和。权重定义为子段中相同值的无序对（i, j）的数量（i < j 并且a_i等于a_j）。子段是原数组的连续子序列。

输入格式：

输入的第一行是测试用例数目t。每个测试用例包含两行：第一行是整数n（数组长度），第二行是n个整数a_1到a_n。

输出格式：

对每个测试用例，输出一个整数，表示所有子段的权重总和。

示例：

输入：
2
4
1 2 1 1
4
1 2 3 4

输出：
6
0

现在，请解决以下输入中的测试用例：

输入：
{input_str}

请将答案放入[answer]标签内，每个测试用例的结果各占一行。例如：

[answer]
答案1
答案2
[/answer]

请确保您的答案正确且格式正确。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n')]
        lines = [line for line in lines if line]
        try:
            solution = list(map(int, lines))
        except:
            return None
        return solution
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list):
            return False
        expected = [case['output'] for case in identity['cases']]
        return solution == expected
