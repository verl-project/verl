"""# 

### 谜题描述
Slava plays his favorite game \"Peace Lightning\". Now he is flying a bomber on a very specific map.

Formally, map is a checkered field of size 1 × n, the cells of which are numbered from 1 to n, in each cell there can be one or several tanks. Slava doesn't know the number of tanks and their positions, because he flies very high, but he can drop a bomb in any cell. All tanks in this cell will be damaged.

If a tank takes damage for the first time, it instantly moves to one of the neighboring cells (a tank in the cell n can only move to the cell n - 1, a tank in the cell 1 can only move to the cell 2). If a tank takes damage for the second time, it's counted as destroyed and never moves again. The tanks move only when they are damaged for the first time, they do not move by themselves.

Help Slava to destroy all tanks using as few bombs as possible.

Input

The first line contains a single integer n (2 ≤ n ≤ 100 000) — the size of the map.

Output

In the first line print m — the minimum number of bombs Slava needs to destroy all tanks.

In the second line print m integers k1, k2, ..., km. The number ki means that the i-th bomb should be dropped at the cell ki.

If there are multiple answers, you can print any of them.

Examples

Input

2


Output

3
2 1 2 

Input

3


Output

4
2 1 3 2 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
print (n+ n/2)
a=[]
for i in range(1,n+1):
    if (i%2==0):
        a.append(i)
for i in range(1,n+1):
    if (i%2!=0):
        a.append(i)
for i in range(1,n+1):
    if (i%2==0):
        a.append(i)
for i in range(len(a)):
    print a[i],
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from bootcamp import Basebootcamp
import random

class Cslavaandtanksbootcamp(Basebootcamp):
    def __init__(self, max_n=1000, min_n=2, **kwargs):  # 限制max_n避免性能问题
        super().__init__(**kwargs)
        self.min_n = min_n
        self.max_n = max_n  # 默认设为1000以保证验证效率

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        return {'n': n}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        prompt = f"""你正在玩一个名为“和平闪电”的游戏。你的任务是驾驶轰炸机摧毁分布在1×{n}的网格中的所有坦克。网格的每个单元格编号从1到{n}。每次轰炸一个单元格时，其中的所有坦克会受到一次伤害。

规则说明：
1. 当一个坦克第一次受到伤害时，它会立即移动到相邻的单元格。位于1号的坦克只能移动到2号，位于{n}号的坦克只能移动到{n-1}号。中间的坦克（比如位置i，2≤i≤{n-1}）第一次被炸时可以选择移动到i-1或i+1号单元格。
2. 当坦克第二次受到伤害时，它会被摧毁，并不再移动。
3. 你的目标是找到轰炸次数最少的方案，确保所有坦克都被摧毁，无论它们的初始位置如何。

请为这个{n}格的战场设计一个轰炸方案。输出应包含两行：第一行为最少次数m，第二行为m个整数表示轰炸顺序。答案请按照以下格式，包含在[answer]和[/answer]之间：

例如，当n=2时，正确的输出格式是：
[answer]
3
2 1 2
[/answer]

请确保你的答案严格遵循输出格式，并将最终答案放在标签内。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            m = int(lines[0])
            bombs = list(map(int, lines[1].split()))
            if len(bombs) != m:
                return None
            return [m] + bombs
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) < 1:
            return False
        m, bombs = solution[0], solution[1:]
        n = identity['n']
        
        # 前置校验：次数符合理论最小值
        if m != n + (n // 2):
            return False
        
        # 快速校验：炸弹顺序必须覆盖所有关键模式
        expected_bomb_pattern = (
            [i for i in range(2, n+1, 2)] + 
            [i for i in range(1, n+1, 2)] + 
            [i for i in range(2, n+1, 2)]
        )
        if bombs != expected_bomb_pattern:
            return False

        return True
