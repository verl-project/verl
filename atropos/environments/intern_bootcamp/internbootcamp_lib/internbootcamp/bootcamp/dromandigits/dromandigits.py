"""# 

### 谜题描述
Let's introduce a number system which is based on a roman digits. There are digits I, V, X, L which correspond to the numbers 1, 5, 10 and 50 respectively. The use of other roman digits is not allowed.

Numbers in this system are written as a sequence of one or more digits. We define the value of the sequence simply as the sum of digits in it.

For example, the number XXXV evaluates to 35 and the number IXI — to 12.

Pay attention to the difference to the traditional roman system — in our system any sequence of digits is valid, moreover the order of digits doesn't matter, for example IX means 11, not 9.

One can notice that this system is ambiguous, and some numbers can be written in many different ways. Your goal is to determine how many distinct integers can be represented by exactly n roman digits I, V, X, L.

Input

The only line of the input file contains a single integer n (1 ≤ n ≤ 10^9) — the number of roman digits to use.

Output

Output a single integer — the number of distinct integers which can be represented using n roman digits exactly.

Examples

Input

1


Output

4


Input

2


Output

10


Input

10


Output

244

Note

In the first sample there are exactly 4 integers which can be represented — I, V, X and L.

In the second sample it is possible to represent integers 2 (II), 6 (VI), 10 (VV), 11 (XI), 15 (XV), 20 (XX), 51 (IL), 55 (VL), 60 (XL) and 100 (LL).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
ff = [1, 5, 10, 50]

\"\"\"1
5
10
50
----------
2      1 1
6      1 5
11     1 10
51     1 50

10     5 5
15     5 10
55     5 50

20     10 10
60     10 50

100    50 50
-----------
\"\"\"



if n == 1:
    print 4
elif n < 12:
    ll = set(ff)
    tut = 0
    for lvl in xrange(1, n):
        temp = set()
        for elem in ll:
            for el in ff:
                temp.add(elem+el)

        tut = len(ll)
        ll = temp
        #print lvl+1, len(ll) - tut, len(ll)

    print len(ll)
else:
    print (n-11)*49 + 292
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from bootcamp import Basebootcamp
import random

class Dromandigitsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10**9):
        """
        初始化参数，确保合理的数值范围
        """
        self.min_n = max(1, min_n)
        self.max_n = max(self.min_n, max_n)  # 修正参数校验逻辑
    
    def case_generator(self):
        """
        生成包含不同特征的测试用例：
        - 小数值（覆盖示例范围）
        - 临界值（n=11,12）
        - 大数值（超过12的随机数）
        """
        # 20%概率生成示例数值
        if random.random() < 0.2:
            return {'n': random.choice([1, 2, 10, 11, 12])}
        # 80%概率生成随机数值
        return {'n': random.randint(self.min_n, self.max_n)}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        return f"""你是一位罗马数字系统专家，请解答以下数学问题：

我们定义一种特殊的数值系统，使用四个罗马字符：
I(1), V(5), X(10), L(50)
数值计算规则为字符对应值的简单相加（忽略传统的位置规则），例如：
- IX → 1+10=11
- VL → 5+50=55
- XX → 10+10=20

问题：当恰好使用{n}个上述字符时，可以表示多少个不同的整数值？

注意事项：
1. 必须使用恰好{n}个字符
2. 不同排列但数值相同的视为同一种表示
3. 最终答案应为唯一的整数结果

请将最终答案放置在[answer]和[/answer]标签之间，例如：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        return solution == cls.calculate_answer(n)

    @staticmethod
    def calculate_answer(n):
        if n == 1:
            return 4
        elif n <= 11:
            values = {1, 5, 10, 50}
            for _ in range(n-1):
                next_gen = set()
                for v in values:
                    next_gen.update([v+1, v+5, v+10, v+50])
                values = next_gen
            return len(values)
        else:
            return 49 * (n - 11) + 292
