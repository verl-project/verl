"""# 

### 谜题描述
Let a and b be two arrays of lengths n and m, respectively, with no elements in common. We can define a new array merge(a,b) of length n+m recursively as follows:

  * If one of the arrays is empty, the result is the other array. That is, merge(∅,b)=b and merge(a,∅)=a. In particular, merge(∅,∅)=∅. 
  * If both arrays are non-empty, and a_1<b_1, then merge(a,b)=[a_1]+merge([a_2,…,a_n],b). That is, we delete the first element a_1 of a, merge the remaining arrays, then add a_1 to the beginning of the result. 
  * If both arrays are non-empty, and a_1>b_1, then merge(a,b)=[b_1]+merge(a,[b_2,…,b_m]). That is, we delete the first element b_1 of b, merge the remaining arrays, then add b_1 to the beginning of the result. 



This algorithm has the nice property that if a and b are sorted, then merge(a,b) will also be sorted. For example, it is used as a subroutine in merge-sort. For this problem, however, we will consider the same procedure acting on non-sorted arrays as well. For example, if a=[3,1] and b=[2,4], then merge(a,b)=[2,3,1,4].

A permutation is an array consisting of n distinct integers from 1 to n in arbitrary order. For example, [2,3,1,5,4] is a permutation, but [1,2,2] is not a permutation (2 appears twice in the array) and [1,3,4] is also not a permutation (n=3 but there is 4 in the array).

There is a permutation p of length 2n. Determine if there exist two arrays a and b, each of length n and with no elements in common, so that p=merge(a,b).

Input

The first line contains a single integer t (1≤ t≤ 1000) — the number of test cases. Next 2t lines contain descriptions of test cases. 

The first line of each test case contains a single integer n (1≤ n≤ 2000).

The second line of each test case contains 2n integers p_1,…,p_{2n} (1≤ p_i≤ 2n). It is guaranteed that p is a permutation.

It is guaranteed that the sum of n across all test cases does not exceed 2000.

Output

For each test case, output \"YES\" if there exist arrays a, b, each of length n and with no common elements, so that p=merge(a,b). Otherwise, output \"NO\".

Example

Input


6
2
2 3 1 4
2
3 1 2 4
4
3 2 6 1 5 7 8 4
3
1 2 3 4 5 6
4
6 1 3 7 4 5 8 2
6
4 3 2 5 1 11 9 12 8 6 10 7


Output


YES
NO
YES
YES
NO
NO

Note

In the first test case, [2,3,1,4]=merge([3,1],[2,4]).

In the second test case, we can show that [3,1,2,4] is not the merge of two arrays of length 2.

In the third test case, [3,2,6,1,5,7,8,4]=merge([3,2,8,4],[6,1,5,7]).

In the fourth test case, [1,2,3,4,5,6]=merge([1,3,6],[2,4,5]), for example.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
_interactive = False

def main():
    for _ in range(int(input())):
        n = int(input())
        ar = input_as_list()

        blocks = []
        prev = ar[0]
        sz = 0
        for x in ar:
            if x > prev:
                prev = x
                blocks += [sz]
                sz = 0
            sz += 1
        blocks += [sz]

        dp = [1] + [0]*n
        for b in blocks:
            for i in reversed(range(n+1)):
                if dp[i] and i+b <= n:
                    dp[i+b] = 1

        print(\"YES\" if dp[n] else \"NO\")


# Constants
INF = float('inf')
MOD = 10**9+7
alphabets = 'abcdefghijklmnopqrstuvwxyz'

# Python3 equivalent names
import os, sys, itertools
if sys.version_info[0] < 3:
    input = raw_input
    range = xrange

    filter = itertools.ifilter
    map = itertools.imap
    zip = itertools.izip

# print-flush in interactive problems
if _interactive:
    flush = sys.stdout.flush
    def printf(*args, **kwargs):
        print(*args, **kwargs)
        flush()

# Debug print, only works on local machine
LOCAL = \"LOCAL_\" in os.environ
debug_print = (print) if LOCAL else (lambda *x, **y: None)

# Fast IO
if (not LOCAL) and (not _interactive):
    from io import BytesIO
    from atexit import register
    sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
    sys.stdout = BytesIO()
    register(lambda: os.write(1, sys.stdout.getvalue()))
    input = lambda: sys.stdin.readline().rstrip('\r\n')

# Some utility functions(Input, N-dimensional lists, ...)
def input_as_list():
    return [int(x) for x in input().split()]

def input_with_offset(o):
    return [int(x)+o for x in input().split()]

def input_as_matrix(n, m):
    return [input_as_list() for _ in range(n)]

def array_of(f, *dim):
    return [array_of(f, *dim[1:]) for _ in range(dim[0])] if dim else f()

main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dunmergebootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.params = {
            'min_n': params.get('min_n', 1),
            'max_n': params.get('max_n', 2000)
        }
    
    def case_generator(self):
        n = random.randint(self.params['min_n'], self.params['max_n'])
        all_elements = list(range(1, 2 * n + 1))
        random.shuffle(all_elements)
        a = all_elements[:n]
        b = all_elements[n:]
        random.shuffle(a)
        random.shuffle(b)
        p = self.merge(a, b)
        return {'n': n, 'p': p}
    
    @staticmethod
    def merge(a, b):
        result = []
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] < b[j]:
                result.append(a[i])
                i += 1
            else:
                result.append(b[j])
                j += 1
        result.extend(a[i:])
        result.extend(b[j:])
        return result
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        p = question_case['p']
        p_str = ' '.join(map(str, p))
        prompt = f"""
        你有一个排列p，长度为2n，其中n={n}。p的具体值为：{p_str}。

        你需要判断是否存在两个长度为n的数组a和b，且它们的元素互不相同，使得p可以通过merge(a,b)过程得到。

        merge(a,b)的定义如下：
        - 如果其中一个数组为空，则结果是另一个数组。
        - 如果两个数组都不为空，比较a的第一个元素和b的第一个元素，较小的放在前面，然后递归处理剩下的部分。

        例如，a=[3,1]，b=[2,4]，则merge(a,b)=[2,3,1,4]。

        请判断是否存在这样的a和b，并将你的答案（YES或NO）放在[answer]标签中。

        请将答案以以下格式输出：
        [answer]
        YES或NO
        [/answer]
        """
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        solution = matches[-1].strip().upper()
        if solution not in ('YES', 'NO'):
            return None
        return solution
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        p = identity['p']
        if not p:
            return False
        prev = p[0]
        sz = 0
        blocks = []
        for x in p:
            if x > prev:
                blocks.append(sz)
                sz = 0
                prev = x
            sz += 1
        blocks.append(sz)
        dp = [False] * (n + 1)
        dp[0] = True
        for b in blocks:
            for i in range(n, -1, -1):
                if dp[i] and (i + b) <= n:
                    if not dp[i + b]:
                        dp[i + b] = True
        expected_answer = 'YES' if dp[n] else 'NO'
        return solution.upper() == expected_answer
