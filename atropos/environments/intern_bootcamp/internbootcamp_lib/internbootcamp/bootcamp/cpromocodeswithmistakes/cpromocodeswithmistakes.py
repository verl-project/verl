"""# 

### 谜题描述
During a New Year special offer the \"Sudislavl Bars\" offered n promo codes. Each promo code consists of exactly six digits and gives right to one free cocktail at the bar \"Mosquito Shelter\". Of course, all the promocodes differ.

As the \"Mosquito Shelter\" opens only at 9, and partying in Sudislavl usually begins at as early as 6, many problems may arise as to how to type a promotional code without errors. It is necessary to calculate such maximum k, that the promotional code could be uniquely identified if it was typed with no more than k errors. At that, k = 0 means that the promotional codes must be entered exactly.

A mistake in this problem should be considered as entering the wrong numbers. For example, value \"123465\" contains two errors relative to promocode \"123456\". Regardless of the number of errors the entered value consists of exactly six digits.

Input

The first line of the output contains number n (1 ≤ n ≤ 1000) — the number of promocodes.

Each of the next n lines contains a single promocode, consisting of exactly 6 digits. It is guaranteed that all the promocodes are distinct. Promocodes can start from digit \"0\".

Output

Print the maximum k (naturally, not exceeding the length of the promocode), such that any promocode can be uniquely identified if it is typed with at most k mistakes.

Examples

Input

2
000000
999999


Output

2


Input

6
211111
212111
222111
111111
112111
121111


Output

0

Note

In the first sample k < 3, so if a bar customer types in value \"090909\", then it will be impossible to define which promocode exactly corresponds to it.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
import pdb
import math
import operator

insys = sys.stdin

def msort(arr, l, r, reverse=True):
    if r - l <= 1:
        return

    mid = (l + r) // 2
    msort(arr, l, mid, reverse=reverse)
    msort(arr, mid, r, reverse=reverse)


    left, right = arr[l:mid], arr[mid:r]
    i, j = 0, 0 
    for it in xrange(l, r):
        if j >= len(right) or (i < len(left) and (left[i] > right[j] if reverse else left[i] < right[j])):
            arr[it] = left[i]
            i += 1
        else:
            arr[it] = right[j]
            j += 1


def gin(func=int):
    if func is None:
        return raw_input().split()

    return map(func, raw_input().split())


def main():
    n = input()

    arr = []
    for code in insys:
        arr.append(code)

    if n == 1:
        return 6

    k = 2
    for i in xrange(n):
        for j in xrange(i + 1, n):
            it = 0
            diff = 0
            while it < 6:
                if arr[i][it] != arr[j][it]:
                    diff += 1
                it += 1
            kt = diff / 2
            if diff % 2 == 0:
                kt -= 1

            if kt < k:
                k = kt
                if k == 0:
                    return 0

    return max(k, 0)

if __name__ == '__main__':
    print main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from itertools import combinations
from bootcamp import Basebootcamp

class Cpromocodeswithmistakesbootcamp(Basebootcamp):
    def __init__(self, n=None, min_n=1, max_n=1000):
        self.n = n if n is not None else random.randint(min_n, max_n)
        if not (1 <= self.n <= 1000):
            raise ValueError("n must be between 1 and 1000")

    def case_generator(self):
        """优化后的高效生成算法"""
        n = self.n  # 使用实例的n值
        codes = set()
        full_range = list(range(10**6))
        random.shuffle(full_range)
        for num in full_range[:n]:
            codes.add(f"{num:06d}")
        return {
            'n': n,
            'promocodes': sorted(list(codes))  # 保证有序
        }

    @staticmethod
    def prompt_func(question_case):
        """修复后的提示生成方法"""
        n = question_case['n']  # 正确获取n值
        codes = "\n".join(question_case['promocodes'])
        return f"""计算促销码最大容错值k。规则：
1. 每个促销码为6位不同数字
2. 错误定义为任意位置数字错误
3. 找出最大k使得输入错误≤k时可唯一确定正确码

当前{n}个促销码：
{codes}

答案格式：[answer]数字[/answer]"""

    @staticmethod
    def extract_output(output):
        """答案提取方法"""
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """验证逻辑优化"""
        codes = identity['promocodes']
        n = identity['n']
        
        if n == 1:
            return solution == 6
        
        min_k = 6  # 初始化为最大可能值
        for a, b in combinations(codes, 2):
            diff = sum(1 for x, y in zip(a, b) if x != y)
            current_k = (diff - 1) // 2
            min_k = min(min_k, current_k)
            if min_k == 0:  # 提前终止条件
                break
        return solution == max(min_k, 0)
