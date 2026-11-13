"""# 

### 谜题描述
Maxim loves to fill in a matrix in a special manner. Here is a pseudocode of filling in a matrix of size (m + 1) × (m + 1):

<image>

Maxim asks you to count, how many numbers m (1 ≤ m ≤ n) are there, such that the sum of values in the cells in the row number m + 1 of the resulting matrix equals t.

Expression (x xor y) means applying the operation of bitwise excluding \"OR\" to numbers x and y. The given operation exists in all modern programming languages. For example, in languages C++ and Java it is represented by character \"^\", in Pascal — by \"xor\".

Input

A single line contains two integers n and t (1 ≤ n, t ≤ 1012, t ≤ n + 1).

Please, do not use the %lld specifier to read or write 64-bit integers in C++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

In a single line print a single integer — the answer to the problem. 

Examples

Input

1 1


Output

1


Input

3 2


Output

1


Input

3 3


Output

0


Input

1000000000000 1048576


Output

118606527258

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
c = []

def f(n, t):
	x = 0
	while 2 ** x <= n:
		x += 1
	x -= 1
	ans = c[x][t + 1] if t + 1 <= x else 0
	if n > 2 ** x and t > 0:
		ans += f(n - 2 ** x, t - 1)
		if t == 1:
			ans += 1
	return ans

c.append([1])
for i in range(1, 60):
	q = [1]
	for j in range(1, i):
		q.append(c[-1][j] + c[-1][j - 1])
	q.append(1)
	c.append(q)

n, t = map(int, raw_input().split())
n += 1
tt = 0
while t % 2 == 0:
	t /= 2
	tt += 1
if t > 1:
	print 0
else:
	print f(n, tt)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random  # 添加缺失的random模块导入
from bootcamp import Basebootcamp

class Cmaximandmatrixbootcamp(Basebootcamp):
    _c = None

    def __init__(self, **params):
        super().__init__(**params)

    @classmethod
    def _build_c(cls):
        if cls._c is not None:
            return
        c = []
        c.append([1])
        for i in range(1, 60):
            q = [1]
            for j in range(1, i):
                q.append(c[i-1][j] + c[i-1][j-1])
            q.append(1)
            c.append(q)
        cls._c = c

    @classmethod
    def f(cls, n, t):
        cls._build_c()
        if n == 0 and t == 0:
            return 1
        if t < 0:
            return 0
        x = 0
        while (2 ** x) <= n:
            x += 1
        x -= 1
        max_k = x
        if (t + 1) <= max_k:
            ans = cls._c[x][t + 1]
        else:
            ans = 0
        remaining = n - (2 ** x)
        if remaining > 0 and t > 0:
            ans += cls.f(remaining, t - 1)
            if t == 1:
                ans += 1
        return ans

    def case_generator(self):
        max_n = 10**12
        # 简化生成逻辑确保数值在合法范围
        n = random.randint(1, max_n)
        t_max = min(n + 1, 10**12)
        t = random.randint(1, t_max)
        return {"n": n, "t": t}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["n"]
        t = question_case["t"]
        prompt = (
            "Maxim loves to fill a matrix in a special way. Given two integers n and t, your task is to count how many integers m (1 ≤ m ≤ n) satisfy the following condition: The sum of values in the (m+1)th row (1-based) of the matrix equals t.\n\n"
            "Matrix Construction Rules:\n"
            "- The matrix is of size (m+1) × (m+1), where rows and columns are 0-based.\n"
            "- Each cell at row i and column j contains the value (i XOR j).\n\n"
            "Input Constraints:\n"
            f"- 1 ≤ n, t ≤ 10^12\n"
            f"- t ≤ n + 1\n\n"
            "Your task is to compute the number of valid m values. Provide your answer as an integer enclosed within [answer] and [/answer] tags.\n\n"
            "Example:\n"
            "Input: 1 1\n"
            "Output: [answer]1[/answer]\n\n"
            f"Now, solve for n = {n} and t = {t}. Place your final answer within [answer] tags."
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n_original = identity["n"]
        t_original = identity["t"]
        n_code = n_original + 1
        t = t_original
        tt = 0
        
        # 处理t必须为2的幂次的条件
        while t % 2 == 0 and t != 0:
            t = t // 2
            tt += 1
        if t != 1:
            correct_answer = 0
        else:
            correct_answer = cls.f(n_code, tt)
        
        try:
            user_answer = int(solution)
            return user_answer == correct_answer
        except:
            return False
