"""# 

### 谜题描述
Volodya and Vlad play the following game. There are k pies at the cells of n × m board. Each turn Volodya moves one pie to the neighbouring (by side) cell. If the pie lies at the border of the board then Volodya can move it outside the board, get the pie and win. After Volodya's move, Vlad bans some edge at the border of the board of length 1 (between two knots of the board) so that Volodya is not able to move the pie outside the board through this edge anymore. The question is: will Volodya win this game? We suppose both players follow the optimal strategy.

<image>

Input

First line contains 3 integers, separated by space: 1 ≤ n, m ≤ 100 — dimensions of the board and 0 ≤ k ≤ 100 — the number of pies. Each of the next k lines contains 2 integers, separated by space: 1 ≤ x ≤ n, 1 ≤ y ≤ m — coordinates of the corresponding pie. There could be more than one pie at a cell. 

Output

Output only one word: \"YES\" — if Volodya wins, \"NO\" — otherwise.

Examples

Input

2 2 1
1 2


Output

YES

Input

3 4 0


Output

NO

Input

100 50 2
50 25
50 25


Output

NO

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m,k=map(int,raw_input().split())
f=\"NO\"
for i in range(1,k+1):
    x,y=map(int,raw_input().split())
    if x<6 or y<6 or n-x<5 or m-y<5:
        f=\"YES\"
print f
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cpieordiebootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=100, m_min=1, m_max=100, k_min=0, k_max=100):
        self.n_min = max(n_min, 1)
        self.n_max = max(n_max, self.n_min)
        self.m_min = max(m_min, 1)
        self.m_max = max(m_max, self.m_min)
        self.k_min = max(k_min, 0)
        self.k_max = max(k_max, self.k_min)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        k = random.randint(self.k_min, self.k_max)
        pies = []
        answer = 'NO'
        
        if k == 0:
            return {'n':n, 'm':m, 'k':k, 'pies':pies, 'answer':answer}
        
        # Calculate safe zone boundaries
        x_safe_min = 6
        x_safe_max = n - 5
        y_safe_min = 6
        y_safe_max = m - 5
        
        can_no = (x_safe_min <= x_safe_max) and (y_safe_min <= y_safe_max)
        
        if can_no and random.random() < 0.5:
            try:
                pies = [
                    (random.randint(x_safe_min, x_safe_max), 
                     random.randint(y_safe_min, y_safe_max))
                    for _ in range(k)
                ]
                answer = 'NO'
            except ValueError:
                can_no = False
        
        if not can_no:
            danger_pies = []
            for _ in range(k):
                if _ == 0 or random.random() < 0.3:
                    edge = random.choice(['top', 'bottom', 'left', 'right'])
                    if edge in ['top', 'bottom']:
                        x = 1 if edge == 'top' else n
                        y = random.randint(1, m)
                    else:
                        x = random.randint(1, n)
                        y = 1 if edge == 'left' else m
                else:
                    x = random.randint(max(1, n-4), n) if random.random() < 0.5 else random.randint(1, min(5, n))
                    y = random.randint(max(1, m-4), m) if random.random() < 0.5 else random.randint(1, min(5, m))
                danger_pies.append((x, y))
            pies = danger_pies
            answer = 'YES'
        
        return {
            'n': n,
            'm': m,
            'k': k,
            'pies': pies,
            'answer': answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        pies = question_case['pies']
        prompt = f"""你是Volodya的教练，需要帮助他判断是否能在与Vlad的游戏中获胜。请仔细分析以下棋局并给出答案。

问题描述：

Volodya和Vlad在一个{n}行{m}列的棋盘上进行游戏。棋盘上共有{k}个派，分布在不同的单元格中。每个回合，Volodya可以选择一个派，将其移动到相邻的单元格（上下左右）。如果派位于棋盘的边缘，Volodya可以将其移出棋盘并立即获胜。之后，Vlad会封锁棋盘边缘的一个单位边，阻止之后通过该边移出派。双方都采取最优策略。请判断Volodya是否能确保胜利。

输入格式：

第一行输入三个整数n m k，分别表示棋盘的行数、列数和派的数量。接下来k行，每行两个整数x y，表示派的位置。

你的任务是根据以下输入数据，判断Volodya是否能够获胜，并输出“YES”或“NO”。

输入数据：
{n} {m} {k}"""
        for x, y in pies:
            prompt += f"\n{x} {y}"
        prompt += "\n\n输出要求：\n请将你的答案置于[answer]标签内，例如：[answer]YES[/answer]或[answer]NO[/answer]。确保答案全部大写，并且是唯一的正确选项。"
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip().upper()
        return last_match if last_match in ['YES', 'NO'] else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answer', 'NO')
