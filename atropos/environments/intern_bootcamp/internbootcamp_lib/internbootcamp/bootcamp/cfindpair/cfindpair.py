"""# 

### 谜题描述
You've got another problem dealing with arrays. Let's consider an arbitrary sequence containing n (not necessarily different) integers a1, a2, ..., an. We are interested in all possible pairs of numbers (ai, aj), (1 ≤ i, j ≤ n). In other words, let's consider all n2 pairs of numbers, picked from the given array.

For example, in sequence a = {3, 1, 5} are 9 pairs of numbers: (3, 3), (3, 1), (3, 5), (1, 3), (1, 1), (1, 5), (5, 3), (5, 1), (5, 5).

Let's sort all resulting pairs lexicographically by non-decreasing. Let us remind you that pair (p1, q1) is lexicographically less than pair (p2, q2) only if either p1 < p2, or p1 = p2 and q1 < q2.

Then the sequence, mentioned above, will be sorted like that: (1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3), (5, 5)

Let's number all the pair in the sorted list from 1 to n2. Your task is formulated like this: you should find the k-th pair in the ordered list of all possible pairs of the array you've been given.

Input

The first line contains two integers n and k (1 ≤ n ≤ 105, 1 ≤ k ≤ n2). The second line contains the array containing n integers a1, a2, ..., an ( - 109 ≤ ai ≤ 109). The numbers in the array can coincide. All numbers are separated with spaces.

Please do not use the %lld specificator to read or write 64-bit integers in С++. It is preferred to use cin, cout, streams or the %I64d specificator instead.

Output

In the single line print two numbers — the sought k-th pair.

Examples

Input

2 4
2 1


Output

2 2


Input

3 2
3 1 5


Output

1 3

Note

In the first sample the sorted sequence for the given array looks as: (1, 1), (1, 2), (2, 1), (2, 2). The 4-th of them is pair (2, 2).

The sorted sequence for the array from the second sample is given in the statement. The 2-nd pair there is (1, 3).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
import traceback
  
def solve():
    n,p = map( int, sys.stdin.readline().strip('\n\r ').split())
    p -= 1
    vs = map( int, sys.stdin.readline().strip('\n\r ').split())
    lenvs = len(vs)
    vs.sort()
    if n==1: return ( vs[p / lenvs ], vs[p % lenvs] )
    prow = p / lenvs
    vrow = vs[prow]
    if prow==0 and vrow<vs[prow+1]: return ( vs[p / lenvs ], vs[p % lenvs] )
    if (prow+1)==n and vrow>vs[prow-1]: return ( vs[p / lenvs ], vs[p % lenvs] )
    if n>2 and prow<(n-1) and prow>0 and vrow>vs[prow-1] and vrow<vs[prow+1]: return ( vs[p / lenvs ], vs[p % lenvs] )

    prow0 = prow
    while prow0>0:
      if vs[prow0-1]<vrow: break
      prow0 -= 1

    prow1 = prow+1
    while prow1<n:
      if vs[prow1]>vrow: break
      prow1 += 1

    dprow10 = prow1 - prow0
    ###sys.stdout.write( str((prow0,prow1,dprow10,)) )
    p -= (prow0 * lenvs)

    pcol = 0
    while p >= dprow10:
      pcol += 1
      p -= dprow10
    return( vrow, vs[pcol] )

if __name__==\"__main__\":
  print( \"%d %d\" % solve() )
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_kth_pair(n, k, array):
    vs = sorted(array)  # 确保排序逻辑正确
    p = k - 1
    lenvs = len(vs)
    
    # 处理极端情况
    if lenvs == 0: return (None, None)
    if lenvs == 1: return (vs[0], vs[0])
    
    # 主计算逻辑
    prow = p // lenvs
    vrow = vs[prow]
    
    # 寻找连续元素块边界
    prow0 = prow
    while prow0 > 0 and vs[prow0-1] == vrow:
        prow0 -= 1
    prow1 = prow + 1
    while prow1 < lenvs and vs[prow1] == vrow:
        prow1 += 1
    
    # 计算有效块尺寸
    block_size = prow1 - prow0
    block_start_index = prow0 * lenvs
    
    # 剩余位置计算
    remaining = p - block_start_index
    if remaining < 0:
        return (vs[p//lenvs], vs[p%lenvs])
    
    # 计算列位置
    col = remaining // block_size
    return (vrow, vs[col])

class Cfindpairbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=5, min_val=-10, max_val=10):
        self.n_min = n_min
        self.n_max = n_max
        self.min_val = min_val
        self.max_val = max_val
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        array = [random.randint(self.min_val, self.max_val) for _ in range(n)]
        max_k = n * n
        # 保证k不超过n^2
        k = random.randint(1, max_k)
        return {
            'n': n,
            'k': k,
            'array': array
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [
            f"{question_case['n']} {question_case['k']}",
            ' '.join(map(str, question_case['array']))
        ]
        input_str = '\n'.join(input_lines)
        prompt = f"""你正在解决一个关于数组有序对的编程问题。根据给定数组和整数k，找出所有可能的有序对按字典序排列后的第k个对。

**详细规则**：
1. 生成所有n²个有序对(ai, aj)，每个元素可重复使用
2. 按字典序排序：(p1,q1) < (p2,q2) 当且仅当 p1 < p2 或 (p1=p2且q1 < q2)
3. 输出排序后的第k个对（从1开始计数）

**输入格式**：
- 第一行两个整数n和k
- 第二行n个整数

**当前测试输入**：
{input_str}

请将最终答案用[answer]和[/answer]标签包裹，例如：[answer]2 3[/answer]。确保只包含答案数值，不要包含其他说明。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 增强匹配鲁棒性，允许任意空白符
        pattern = r'\[answer\]\s*(-?\d+)\s+(-?\d+)\s*\[/answer\]'
        matches = re.findall(pattern, output)
        if not matches:
            return None
        last_match = matches[-1]
        try:
            return (int(last_match[0]), int(last_match[1]))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            k = identity['k']
            array = identity['array']
            expected = compute_kth_pair(n, k, array)
            return solution == expected
        except Exception as e:
            return False
