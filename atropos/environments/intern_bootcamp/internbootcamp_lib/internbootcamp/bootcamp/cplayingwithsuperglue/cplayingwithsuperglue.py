"""# 

### 谜题描述
Two players play a game. The game is played on a rectangular board with n × m squares. At the beginning of the game two different squares of the board have two chips. The first player's goal is to shift the chips to the same square. The second player aims to stop the first one with a tube of superglue.

We'll describe the rules of the game in more detail.

The players move in turns. The first player begins.

With every move the first player chooses one of his unglued chips, and shifts it one square to the left, to the right, up or down. It is not allowed to move a chip beyond the board edge. At the beginning of a turn some squares of the board may be covered with a glue. The first player can move the chip to such square, in this case the chip gets tightly glued and cannot move any longer.

At each move the second player selects one of the free squares (which do not contain a chip or a glue) and covers it with superglue. The glue dries long and squares covered with it remain sticky up to the end of the game.

If, after some move of the first player both chips are in the same square, then the first player wins. If the first player cannot make a move (both of his chips are glued), then the second player wins. Note that the situation where the second player cannot make a move is impossible — he can always spread the glue on the square from which the first player has just moved the chip.

We will further clarify the case where both chips are glued and are in the same square. In this case the first player wins as the game ends as soon as both chips are in the same square, and the condition of the loss (the inability to move) does not arise.

You know the board sizes and the positions of the two chips on it. At the beginning of the game all board squares are glue-free. Find out who wins if the players play optimally.

Input

The first line contains six integers n, m, x1, y1, x2, y2 — the board sizes and the coordinates of the first and second chips, correspondingly (1 ≤ n, m ≤ 100; 2 ≤ n × m; 1 ≤ x1, x2 ≤ n; 1 ≤ y1, y2 ≤ m). The numbers in the line are separated by single spaces.

It is guaranteed that the chips are located in different squares.

Output

If the first player wins, print \"First\" without the quotes. Otherwise, print \"Second\" without the quotes.

Examples

Input

1 6 1 2 1 6


Output

First

Input

6 5 4 3 2 1


Output

First

Input

10 10 1 1 10 10


Output

Second

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m,x1,y1,x2,y2=map(int,raw_input().split())
if abs(x1-x2)>abs(y1-y2):
    x1,y1=y1,x1
    x2,y2=y2,x2
win=(abs(y1-y2)<=4)and(abs(x1-x2)+abs(y1-y2)<=6)
print \"First\" if win else \"Second\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cplayingwithsupergluebootcamp(Basebootcamp):
    def __init__(self, n_range=(1, 100), m_range=(1, 100)):
        """
        参数增强校验逻辑，确保合法棋盘范围
        """
        self.n_min = max(1, n_range[0])
        self.n_max = max(1, n_range[1])
        self.m_min = max(1, m_range[0])
        self.m_max = max(1, m_range[1])
        assert self.n_min <= self.n_max and self.m_min <= self.m_max, "Invalid grid size range"
    
    def case_generator(self):
        """
        完全随机生成合法案例，时间复杂度优化为O(1)
        """
        while True:
            n = random.randint(self.n_min, self.n_max)
            m = random.randint(self.m_min, self.m_max)
            if n * m >= 2:  # 严格满足题目约束
                break
        
        # 生成第一组坐标
        x1, y1 = random.randint(1, n), random.randint(1, m)
        
        # 高效生成第二组不重复坐标
        while True:
            x2, y2 = random.randint(1, n), random.randint(1, m)
            if (x1, y1) != (x2, y2):
                break
        
        return {
            'n': n, 'm': m,
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2
        }
    
    @staticmethod
    def prompt_func(question_case):
        """
        增强规则描述的完整性，包含所有必要细节
        """
        rule_desc = [
            "1. Players alternate turns, starting with First",
            "2. First moves one unglued chip (L/R/U/D), can enter glued square (immobilizes chip)",
            "3. Second places glue on an empty square each turn",
            "4. First wins if chips meet after any move",
            "5. Second wins if both chips are immobilized without meeting"
        ]
        return f"""Analyze this chip movement game (Board: {question_case['n']}x{question_case['m']}, Chips at ({question_case['x1']},{question_case['y1']}) & ({question_case['x2']},{question_case['y2']})).

Rules:
{chr(10).join(rule_desc)}

Determine the winner with optimal play. Answer strictly as [answer]First[/answer] or [answer]Second[/answer]."""

    @staticmethod
    def extract_output(output):
        """
        增强抽取鲁棒性，忽略大小写匹配
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.IGNORECASE)
        if not matches:
            return None
        last_ans = matches[-1].strip().capitalize()
        return last_ans if last_ans in ('First', 'Second') else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        精确复现参考算法的验证逻辑
        """
        # 提取坐标差
        dx = abs(identity['x1'] - identity['x2'])
        dy = abs(identity['y1'] - identity['y2'])
        
        # 按参考算法逻辑处理坐标交换
        if dx > dy:
            dx, dy = dy, dx  # 交换坐标差
        
        # 严格应用判定条件
        is_first_win = (dy <= 4) and (dx + dy <= 6)
        return str(solution).capitalize() == ("First" if is_first_win else "Second")
