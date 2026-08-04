"""# 

### 谜题描述
Sereja has a sequence that consists of n positive integers, a1, a2, ..., an. 

First Sereja took a piece of squared paper and wrote all distinct non-empty non-decreasing subsequences of sequence a. Then for each sequence written on the squared paper, Sereja wrote on a piece of lines paper all sequences that do not exceed it.

A sequence of positive integers x = x1, x2, ..., xr doesn't exceed a sequence of positive integers y = y1, y2, ..., yr, if the following inequation holds: x1 ≤ y1, x2 ≤ y2, ..., xr ≤ yr.

Now Sereja wonders, how many sequences are written on the lines piece of paper. Help Sereja, find the required quantity modulo 1000000007 (109 + 7). 

Input

The first line contains integer n (1 ≤ n ≤ 105). The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 106).

Output

In the single line print the answer to the problem modulo 1000000007 (109 + 7).

Examples

Input

1
42


Output

42


Input

3
1 2 2


Output

13


Input

5
1 2 3 4 5


Output

719

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
def main():
    n = int(stdin.readline())
    a = map(int, stdin.readline().split())
    b = [0] * (1 << 20)
    def get(x):
        res = 0
        while x > 0:
            res += b[x]
            x &= (x - 1)
        return res
    def add(x, v):
        while x < 1 << 20:
            b[x] += v
            x += x - (x & (x - 1))
    mod = 1000000007
    for x in a:
        y = get(x) * x + x
        s = get(x) - get(x - 1)
        add(x, ((y - s) % mod + mod) % mod)
    print get(10 ** 6) % mod
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

MOD = 10**9 + 7
MAX_BIT = 20  # 使用20位二进制数表示树状数组大小
TREE_SIZE = 1 << MAX_BIT  # 1048576，覆盖题目最大数值1e6

class Cserejaandsubsequencesbootcamp(Basebootcamp):
    def __init__(self, max_n=100, max_value=1000):
        self.max_n = min(max_n, 10**5)     # 题目约束n≤1e5
        self.max_value = min(max_value, 10**6)  # 题目约束ai≤1e6
    
    def case_generator(self):
        """生成符合题目数值范围的测试用例"""
        n = random.randint(1, self.max_n)
        a = [random.randint(1, self.max_value) for _ in range(n)]
        return {
            'n': n,
            'a': a,
            'correct_answer': self.compute_answer(n, a)
        }
    
    @staticmethod
    def compute_answer(n, a):
        """优化后的正确答案计算"""
        tree = [0] * (TREE_SIZE + 1)  # 固定大小的树状数组

        def lowbit(x): return x & -x
        
        def get(x):
            res = 0
            while x > 0:
                res = (res + tree[x]) % MOD
                x -= lowbit(x)
            return res

        def update(x, v):
            while x <= TREE_SIZE:
                tree[x] = (tree[x] + v) % MOD
                x += lowbit(x)

        for num in a:
            prefix_sum = get(num)
            # 计算新增值并更新树状数组
            new_val = (prefix_sum * num + num) % MOD
            current = (get(num) - get(num-1)) % MOD  # 获取当前值
            delta = (new_val - current) % MOD
            update(num, delta)

        return get(TREE_SIZE) % MOD  # 查询最大值范围内的总和

    @staticmethod
    def prompt_func(question_case):
        a_str = ' '.join(map(str, question_case['a']))
        return f"""给定一个长度为{question_case['n']}的正整数序列：[{a_str}]
        
请按以下规则计算结果：
1. 找出所有不同的非空非递减子序列y
2. 对每个y，统计满足∀i (x_i ≤ y_i)的非空序列x的数量
3. 将结果求和并对1e9+7取模

答案格式要求：将最终答案用[answer]标签包裹，例如：[answer]123[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) % MOD if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['correct_answer'] % MOD
        return solution == expected if solution is not None else False
