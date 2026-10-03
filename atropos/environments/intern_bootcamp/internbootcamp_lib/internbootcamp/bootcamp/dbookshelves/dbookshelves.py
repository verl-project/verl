"""# 

### 谜题描述
Mr Keks is a typical white-collar in Byteland.

He has a bookshelf in his office with some books on it, each book has an integer positive price.

Mr Keks defines the value of a shelf as the sum of books prices on it. 

Miraculously, Mr Keks was promoted and now he is moving into a new office.

He learned that in the new office he will have not a single bookshelf, but exactly k bookshelves. He decided that the beauty of the k shelves is the [bitwise AND](https://en.wikipedia.org/wiki/Bitwise_operation#AND) of the values of all the shelves.

He also decided that he won't spend time on reordering the books, so he will place several first books on the first shelf, several next books on the next shelf and so on. Of course, he will place at least one book on each shelf. This way he will put all his books on k shelves in such a way that the beauty of the shelves is as large as possible. Compute this maximum possible beauty.

Input

The first line contains two integers n and k (1 ≤ k ≤ n ≤ 50) — the number of books and the number of shelves in the new office.

The second line contains n integers a_1, a_2, … a_n, (0 < a_i < 2^{50}) — the prices of the books in the order they stand on the old shelf.

Output

Print the maximum possible beauty of k shelves in the new office.

Examples

Input

10 4
9 14 28 1 7 13 15 29 2 31


Output

24


Input

7 3
3 14 15 92 65 35 89


Output

64

Note

In the first example you can split the books as follows:

$$$(9 + 14 + 28 + 1 + 7) \& (13 + 15) \& (29 + 2) \& (31) = 24.$$$

In the second example you can split the books as follows:

$$$(3 + 14 + 15 + 92) \& (65) \& (35 + 89) = 64.$$$

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin, stdout
from collections import Counter, defaultdict
from itertools import permutations, combinations
raw_input = stdin.readline
pr = stdout.write


def in_arr():
    return map(int,raw_input().split())


def pr_num(n):
    stdout.write(str(n)+'\n')


def pr_arr(arr):
    for i in arr:
        stdout.write(str(i)+' ')
    stdout.write('\n')


range = xrange # not for python 3.0+
n,k=in_arr()
pre=[0]*(n+1)

l=in_arr()
for i in range(1,n+1):
    pre[i]+=pre[i-1]
    pre[i]+=l[i-1]
ans=0
for bit in range(60,-1,-1):
    tar=ans|(1<<bit)
    dp=[[0 for i in range(n+1)] for i in range(k+1)]
    dp[0][0]=1
    for j in range(1,k+1):
        for k2 in range(1,n+1):
            for k1 in range(k2):
                if dp[j-1][k1] and (pre[k2]-pre[k1])&tar==tar:
                    dp[j][k2]=1
    if dp[k][n]:
        ans=tar
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Dbookshelvesbootcamp(Basebootcamp):
    def __init__(self, max_n=50, max_book_value=(1 << 50)-1):
        self.max_n = max_n
        self.max_book_value = max_book_value
    
    def case_generator(self):
        """生成完全随机且符合题目约束的测试用例"""
        while True:  # 确保至少存在有效分割
            n = random.randint(1, self.max_n)
            k = random.randint(1, n)
            a = [random.randint(1, self.max_book_value-1) for _ in range(n)]
            if self.validate_case(n, k, a):
                return {"n": n, "k": k, "a": a}

    def validate_case(self, n, k, a):
        """验证生成的案例至少存在一个有效分割"""
        try:
            self.compute_max_beauty(n, k, a)
            return True
        except:
            return False
    
    @staticmethod
    def compute_max_beauty(n, k, a):
        """优化后的正确性验证算法"""
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + a[i]

        result = 0
        for bit in reversed(range(61)):
            mask = result | (1 << bit)
            dp = [False] * (n + 1)
            dp[0] = True
            
            for _ in range(k):
                new_dp = [False] * (n + 1)
                for end in range(n+1):
                    if not dp[end]: continue
                    for new_end in range(end+1, n+1):
                        if (prefix[new_end] - prefix[end]) & mask == mask:
                            new_dp[new_end] = True
                dp = new_dp
            
            if dp[n]:
                result = mask
        return result
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["n"]
        k = question_case["k"]
        a = question_case["a"]
        return f"""## 书架美丽值最大化问题

Mr. Keks需要将{n}本价格分别为{a}的书分配到{k}个连续的书架上。每个书架必须包含至少一本书，书架的价值是其上所有书的价格之和。总美丽值是所有书架价值的按位与运算结果。

**任务**：找出能获得最大美丽值的分法。

**输出要求**：将最终答案包裹在[answer]标签中，例如：[answer]42[/answer]。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip().split()[0])
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """带缓存的高效验证"""
        try:
            return solution == cls.compute_max_beauty(
                identity["n"],
                identity["k"],
                identity["a"]
            )
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False
