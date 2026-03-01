"""# 

### 谜题描述
Polycarp has n dice d1, d2, ..., dn. The i-th dice shows numbers from 1 to di. Polycarp rolled all the dice and the sum of numbers they showed is A. Agrippina didn't see which dice showed what number, she knows only the sum A and the values d1, d2, ..., dn. However, she finds it enough to make a series of statements of the following type: dice i couldn't show number r. For example, if Polycarp had two six-faced dice and the total sum is A = 11, then Agrippina can state that each of the two dice couldn't show a value less than five (otherwise, the remaining dice must have a value of at least seven, which is impossible).

For each dice find the number of values for which it can be guaranteed that the dice couldn't show these values if the sum of the shown values is A.

Input

The first line contains two integers n, A (1 ≤ n ≤ 2·105, n ≤ A ≤ s) — the number of dice and the sum of shown values where s = d1 + d2 + ... + dn.

The second line contains n integers d1, d2, ..., dn (1 ≤ di ≤ 106), where di is the maximum value that the i-th dice can show.

Output

Print n integers b1, b2, ..., bn, where bi is the number of values for which it is guaranteed that the i-th dice couldn't show them.

Examples

Input

2 8
4 4


Output

3 3 

Input

1 3
5


Output

4 

Input

2 3
2 3


Output

0 1 

Note

In the first sample from the statement A equal to 8 could be obtained in the only case when both the first and the second dice show 4. Correspondingly, both dice couldn't show values 1, 2 or 3.

In the second sample from the statement A equal to 3 could be obtained when the single dice shows 3. Correspondingly, it couldn't show 1, 2, 4 or 5.

In the third sample from the statement A equal to 3 could be obtained when one dice shows 1 and the other dice shows 2. That's why the first dice doesn't have any values it couldn't show and the second dice couldn't show 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, A = map(int, raw_input().split())
d =  map(int, raw_input().split())
sum=0
for i in range(0,n):
	sum+=d[i]
for i in range(0,n):
	sum-=d[i]
	print max(0,A-sum-1)+max(0,d[i]-(A-(n-1)) ),
	sum+=d[i]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cpolycarpusdicebootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_d=10, **params):
        """
        初始化骰子谜题训练场参数
        :param max_n: 最大骰子数量，默认5
        :param max_d: 单个骰子最大面数，默认10
        """
        super().__init__(**params)  # 关键修复：显式调用父类初始化
        self.max_n = max_n
        self.max_d = max_d

    def case_generator(self):
        """
        生成有效的谜题实例
        返回包含n, A, d的字典
        """
        n = random.randint(1, self.max_n)
        d = [random.randint(1, self.max_d) for _ in range(n)]
        sum_d = sum(d)
        A = random.randint(n, sum_d)
        return {'n': n, 'A': A, 'd': d}

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = f"{question_case['n']} {question_case['A']}\n{' '.join(map(str, question_case['d']))}"
        return f"""Polycarp has rolled some dice and now needs your help to determine impossible values. 

**Problem Statement:**
Given n dice where each dice i can show numbers between 1 and d_i, and the total sum is A, find for each dice how many numbers it **could not possibly** have shown based on these constraints.

**Input Format:**
- First line: n and A (space separated)
- Second line: d_1 d_2 ... d_n (space separated)

**Output Format:**
- n space-separated integers indicating the count of impossible values for each dice

**Example Input/Output:**
Input:
2 8
4 4
Output:
3 3

**Your Task:**
Solve the following case and enclose your answer in [answer]...[/answer] tags.

Input:
{input_lines}

[answer]
Place your answer here (e.g., '1 2 3')
[/answer]
"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return list(map(int, matches[-1].strip().split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            A = identity['A']
            d = identity['d']
            sum_total = sum(d)
            correct = []
            for i in range(n):
                current = d[i]
                sum_others = sum_total - current
                part1 = max(0, A - sum_others - 1)
                part2 = max(0, current - (A - (n-1)))
                correct.append(part1 + part2)
            
            return solution == correct
        except:
            return False
