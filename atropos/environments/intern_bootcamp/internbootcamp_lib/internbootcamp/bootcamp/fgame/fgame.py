"""# 

### 谜题描述
Allen and Bessie are playing a simple number game. They both know a function f: \{0, 1\}^n → R, i. e. the function takes n binary arguments and returns a real value. At the start of the game, the variables x_1, x_2, ..., x_n are all set to -1. Each round, with equal probability, one of Allen or Bessie gets to make a move. A move consists of picking an i such that x_i = -1 and either setting x_i → 0 or x_i → 1.

After n rounds all variables are set, and the game value resolves to f(x_1, x_2, ..., x_n). Allen wants to maximize the game value, and Bessie wants to minimize it.

Your goal is to help Allen and Bessie find the expected game value! They will play r+1 times though, so between each game, exactly one value of f changes. In other words, between rounds i and i+1 for 1 ≤ i ≤ r, f(z_1, ..., z_n) → g_i for some (z_1, ..., z_n) ∈ \{0, 1\}^n. You are to find the expected game value in the beginning and after each change.

Input

The first line contains two integers n and r (1 ≤ n ≤ 18, 0 ≤ r ≤ 2^{18}).

The next line contains 2^n integers c_0, c_1, ..., c_{2^n-1} (0 ≤ c_i ≤ 10^9), denoting the initial values of f. More specifically, f(x_0, x_1, ..., x_{n-1}) = c_x, if x = \overline{x_{n-1} … x_0} in binary.

Each of the next r lines contains two integers z and g (0 ≤ z ≤ 2^n - 1, 0 ≤ g ≤ 10^9). If z = \overline{z_{n-1} ... z_0} in binary, then this means to set f(z_0, ..., z_{n-1}) → g.

Output

Print r+1 lines, the i-th of which denotes the value of the game f during the i-th round. Your answer must have absolute or relative error within 10^{-6}.

Formally, let your answer be a, and the jury's answer be b. Your answer is considered correct if \frac{|a - b|}{max{(1, |b|)}} ≤ 10^{-6}.

Examples

Input

2 2
0 1 2 3
2 5
0 4


Output

1.500000
2.250000
3.250000


Input

1 0
2 3


Output

2.500000


Input

2 0
1 1 1 1


Output

1.000000

Note

Consider the second test case. If Allen goes first, he will set x_1 → 1, so the final value will be 3. If Bessie goes first, then she will set x_1 → 0 so the final value will be 2. Thus the answer is 2.5.

In the third test case, the game value will always be 1 regardless of Allen and Bessie's play.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,r = map(int, raw_input().split())
ls = map(int, raw_input().split())
sm=sum(ls)
nls=2**n
print float(sm)/nls
for i in range(r):
    a,b = map(int, raw_input().split())
    sm+=b-ls[a]
    ls[a]=b
    print float(sm)/nls
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Fgamebootcamp(Basebootcamp):
    def __init__(self, max_n=3, max_r=5, **kwargs):
        """
        初始化训练场参数。
        :param max_n: 最大n值，用于生成测试案例时限制n的范围。
        :param max_r: 最大r值，限制修改次数。
        """
        super().__init__(**kwargs)
        self.max_n = max_n
        self.max_r = max_r

    def case_generator(self):
        """
        生成谜题实例。
        """
        n = random.randint(1, self.max_n)
        size = 2 ** n
        initial = [random.randint(0, 100) for _ in range(size)]
        r = random.randint(0, self.max_r)
        updates = []
        for _ in range(r):
            z = random.randint(0, size - 1)
            g = random.randint(0, 100)
            updates.append({"z": z, "g": g})
        return {
            "n": n,
            "r": r,
            "initial": initial,
            "updates": updates
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        将谜题实例转换为问题描述文本。
        """
        n = question_case["n"]
        r = question_case["r"]
        initial = " ".join(map(str, question_case["initial"]))
        updates = "\n".join([f"{u['z']} {u['g']}" for u in question_case["updates"]])
        example_input = f"{n} {r}\n{initial}"
        if r > 0:
            example_input += "\n" + updates

        prompt = f"""Allen和Bessie正在玩一个数字游戏。已知函数f接受n个二进制参数并返回实数值。游戏开始后，两人轮流随机设置变量的值，最终计算f的值。你的任务是计算游戏开始时和每次函数值变化后的期望值。

输入格式：
第一行是n和r（0 ≤ r ≤ 2^18），第二行有2^n个整数表示初始的f值，接着r行每行给出一个修改(z, g)表示将f在z处的值改为g。

输出格式：
输出r+1行，每行为对应阶段游戏的期望值，保留六位小数。

例如，给定输入：
2 2
0 1 2 3
2 5
0 4

正确输出：
1.500000
2.250000
3.250000

请将答案包含在[answer]标签内，每行一个结果，格式如下：
[answer]
1.500000
2.250000
3.250000
[/answer]

请输入以下测试案例的答案：
{example_input}"""
        return prompt

    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取答案。
        """
        answer_pattern = re.compile(r'\[answer\](.*?)\[/answer\]', re.DOTALL)
        matches = answer_pattern.findall(output)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        solutions = []
        for line in last_answer.split('\n'):
            line = line.strip()
            if line:
                try:
                    solutions.append(float(line))
                except ValueError:
                    continue
        return solutions if solutions else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案是否正确。
        """
        if not solution or len(solution) != identity['r'] + 1:
            return False

        n = identity['n']
        initial = identity['initial'].copy()
        updates = identity['updates']
        total = sum(initial)
        expected = [total / (2 ** n)]
        current_values = initial.copy()

        for update in updates:
            z = update['z']
            g = update['g']
            total += g - current_values[z]
            current_values[z] = g
            expected.append(total / (2 ** n))

        if len(solution) != len(expected):
            return False

        for s, e in zip(solution, expected):
            if abs(s - e) > 1e-6 and abs(s - e) / max(1, abs(e)) > 1e-6:
                return False
        return True
