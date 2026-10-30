"""# 

### 谜题描述
Famous Brazil city Rio de Janeiro holds a tennis tournament and Ostap Bender doesn't want to miss this event. There will be n players participating, and the tournament will follow knockout rules from the very first game. That means, that if someone loses a game he leaves the tournament immediately.

Organizers are still arranging tournament grid (i.e. the order games will happen and who is going to play with whom) but they have already fixed one rule: two players can play against each other only if the number of games one of them has already played differs by no more than one from the number of games the other one has already played. Of course, both players had to win all their games in order to continue participating in the tournament.

Tournament hasn't started yet so the audience is a bit bored. Ostap decided to find out what is the maximum number of games the winner of the tournament can take part in (assuming the rule above is used). However, it is unlikely he can deal with this problem without your help.

Input

The only line of the input contains a single integer n (2 ≤ n ≤ 1018) — the number of players to participate in the tournament.

Output

Print the maximum number of games in which the winner of the tournament can take part.

Examples

Input

2


Output

1


Input

3


Output

2


Input

4


Output

2


Input

10


Output

4

Note

In all samples we consider that player number 1 is the winner.

In the first sample, there would be only one game so the answer is 1.

In the second sample, player 1 can consequently beat players 2 and 3. 

In the third sample, player 1 can't play with each other player as after he plays with players 2 and 3 he can't play against player 4, as he has 0 games played, while player 1 already played 2. Thus, the answer is 2 and to achieve we make pairs (1, 2) and (3, 4) and then clash the winners.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math
n = int(raw_input())
def fib(n):
    if n==0:
        return 1
    if n==1:
        return 1
    a,b = 1,1
    for i in range(1,n):
        a,b = a+b,a
    return a
for i in range(1,100):
    if fib(i)>n:
        break
print i-2
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Atennischampionshipbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        初始化训练场类，设置生成问题实例的参数。
        参数:
            min_n (int): 生成问题的最小玩家数，默认为2。
            max_n (int): 生成问题的最大玩家数，默认为10^18。
        """
        self.params = params
        self.params.setdefault('min_n', 2)
        self.params.setdefault('max_n', 10**18)

    @staticmethod
    def compute_max_games(n):
        """
        计算冠军最多能参加的比赛数。
        参数:
            n (int): 玩家人数。
        返回:
            int: 最大比赛数。
        """
        if n <= 1:
            return 0
        # 使用二分查找找到最大的斐波那契数 F(k+2) <= n
        # 初始化斐波那契数列
        fib_prev, fib_curr = 1, 1  # F(1)=1, F(2)=1
        k = 0
        while True:
            fib_next = fib_prev + fib_curr
            if fib_next > n:
                break
            fib_prev, fib_curr = fib_curr, fib_next
            k += 1
        return k

    def case_generator(self):
        """
        生成谜题实例，包括n和正确答案。
        返回:
            dict: 包含n和max_games的问题实例。
        """
        n = random.randint(self.params['min_n'], self.params['max_n'])
        max_games = self.compute_max_games(n)
        return {'n': n, 'max_games': max_games}

    @staticmethod
    def prompt_func(question_case):
        """
        将问题实例转换为文本问题。
        参数:
            question_case (dict): 包含n的问题实例。
        返回:
            str: 问题字符串。
        """
        n = question_case['n']
        return f"给定n={n}个球员，按照锦标赛规则，找出冠军最多能参加多少场比赛。每场比赛的对手必须满足两人之前打的比赛次数之差不超过1。假设冠军赢了所有比赛。请将答案放在[answer]和[/answer]之间。例如，当n=2时，答案是1。现在，n={n}，请给出答案。"

    @staticmethod
    def extract_output(output):
        """
        从输出中提取答案。
        参数:
            output (str): LLM的完整输出。
        返回:
            int或None: 提取的答案或None。
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            return int(last_answer)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案是否正确。
        参数:
            solution (int): 提取的答案。
            identity (dict): 包含正确答案的问题实例。
        返回:
            bool: 答案是否正确。
        """
        expected = identity['max_games']
        return solution == expected

# 示例使用
if __name__ == "__main__":
    bootcamp = Atennischampionshipbootcamp()
    case = bootcamp.case_generator()
    prompt = Atennischampionshipbootcamp.prompt_func(case)
    print("生成的问题实例:", case)
    print("生成的问题提示:", prompt)
    # 模拟LLM的响应
    response = "冠军最多可以参加 [answer]4[/answer] 场比赛。"
    extracted = Atennischampionshipbootcamp.extract_output(prompt + response)
    print("提取的答案:", extracted)
    score = Atennischampionshipbootcamp.verify_score(extracted, case)
    print("验证得分:", score)
