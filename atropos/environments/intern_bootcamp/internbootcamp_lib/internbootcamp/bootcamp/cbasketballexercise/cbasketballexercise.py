"""# 

### 谜题描述
Finally, a basketball court has been opened in SIS, so Demid has decided to hold a basketball exercise session. 2 ⋅ n students have come to Demid's exercise session, and he lined up them into two rows of the same size (there are exactly n people in each row). Students are numbered from 1 to n in each row in order from left to right.

<image>

Now Demid wants to choose a team to play basketball. He will choose players from left to right, and the index of each chosen player (excluding the first one taken) will be strictly greater than the index of the previously chosen player. To avoid giving preference to one of the rows, Demid chooses students in such a way that no consecutive chosen students belong to the same row. The first student can be chosen among all 2n students (there are no additional constraints), and a team can consist of any number of students. 

Demid thinks, that in order to compose a perfect team, he should choose students in such a way, that the total height of all chosen students is maximum possible. Help Demid to find the maximum possible total height of players in a team he can choose.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 10^5) — the number of students in each row.

The second line of the input contains n integers h_{1, 1}, h_{1, 2}, …, h_{1, n} (1 ≤ h_{1, i} ≤ 10^9), where h_{1, i} is the height of the i-th student in the first row.

The third line of the input contains n integers h_{2, 1}, h_{2, 2}, …, h_{2, n} (1 ≤ h_{2, i} ≤ 10^9), where h_{2, i} is the height of the i-th student in the second row.

Output

Print a single integer — the maximum possible total height of players in a team Demid can choose.

Examples

Input


5
9 3 5 7 3
5 8 1 4 5


Output


29


Input


3
1 2 9
10 1 1


Output


19


Input


1
7
4


Output


7

Note

In the first example Demid can choose the following team as follows: 

<image>

In the second example Demid can choose the following team as follows: 

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
h1 = list(map(int, raw_input().split()))
h2 = list(map(int, raw_input().split()))
dp1 = [0] * n
dp2 = [0] * n
dp1[0] = h1[0]
dp2[0] = h2[0]
for i in range(1, n):
    dp1[i] = max(dp2[i-1] + h1[i], dp1[i-1])
    dp2[i] = max(dp1[i-1] + h2[i], dp2[i-1])
print(max(dp1[n-1], dp2[n-1]))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cbasketballexercisebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 1)
        self.n_max = params.get('n_max', 10)
        self.h_min = params.get('h_min', 1)
        self.h_max = params.get('h_max', 10**9)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        h1 = [random.randint(self.h_min, self.h_max) for _ in range(n)]
        h2 = [random.randint(self.h_min, self.h_max) for _ in range(n)]
        return {
            'n': n,
            'h1': h1,
            'h2': h2
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        h1 = question_case['h1']
        h2 = question_case['h2']
        prompt = (
            f"现在有两排学生，每排有{n}个学生。第一排的身高依次为：{h1}。第二排的身高依次为：{h2}。\n"
            "根据规则，Demid需要选择一个队，使得总身高最大。规则如下：\n"
            "1. 每次选择的学生必须来自另一排，不能连续选同一排。\n"
            "2. 学生的索引必须严格递增，即只能选择当前选中的学生右边的学生。\n"
            "3. 第一个选的学生可以是任意一排的任意位置。\n"
            "请计算 Demid 能选择的最大总身高，并将答案放在[answer]标签中，例如：\n"
            "[answer]29[/answer]\n"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output)
        if not matches:
            return None
        answer_str = matches[-1].strip()
        try:
            return int(answer_str)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        h1 = identity['h1']
        h2 = identity['h2']
        
        if n == 0:
            correct = 0
        else:
            dp1 = [0] * n
            dp2 = [0] * n
            dp1[0] = h1[0]
            dp2[0] = h2[0]
            for i in range(1, n):
                dp1[i] = max(dp2[i-1] + h1[i], dp1[i-1])
                dp2[i] = max(dp1[i-1] + h2[i], dp2[i-1])
            correct = max(dp1[-1], dp2[-1])
        
        return solution == correct
