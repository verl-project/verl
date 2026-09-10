"""# 

### 谜题描述
A sweet little monster Om Nom loves candies very much. One day he found himself in a rather tricky situation that required him to think a bit in order to enjoy candies the most. Would you succeed with the same task if you were on his place?

<image>

One day, when he came to his friend Evan, Om Nom didn't find him at home but he found two bags with candies. The first was full of blue candies and the second bag was full of red candies. Om Nom knows that each red candy weighs Wr grams and each blue candy weighs Wb grams. Eating a single red candy gives Om Nom Hr joy units and eating a single blue candy gives Om Nom Hb joy units.

Candies are the most important thing in the world, but on the other hand overeating is not good. Om Nom knows if he eats more than C grams of candies, he will get sick. Om Nom thinks that it isn't proper to leave candy leftovers, so he can only eat a whole candy. Om Nom is a great mathematician and he quickly determined how many candies of what type he should eat in order to get the maximum number of joy units. Can you repeat his achievement? You can assume that each bag contains more candies that Om Nom can eat.

Input

The single line contains five integers C, Hr, Hb, Wr, Wb (1 ≤ C, Hr, Hb, Wr, Wb ≤ 109).

Output

Print a single integer — the maximum number of joy units that Om Nom can get.

Examples

Input

10 3 5 2 3


Output

16

Note

In the sample test Om Nom can eat two candies of each type and thus get 16 joy units.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division

c, p, q, r, s = map(int, raw_input().split())
inf = float('inf')

def solve1(c, a, b, L, U):
    U = min(U, b // a)
    if c >= 0:
        return U
    else:
        return max(0, L)            

def solve(c1, c2, a1, a2, b, L, U):
    if b < 0:
        return (-inf, -inf)
    def value((x, y)):
        return c1 * x + c2 * y
    if c1 <= 0:
        return (L, solve1(c2, a2, b - a1 * L, 0, inf))
    elif c2 <= 0:
        return (solve1(c1, a1, b, L, U), 0)
    elif a1 == 0:
        return (U, solve1(c2, a2, b, 0, inf))
    elif a2 == 0:
        return (0, inf)
    elif L == U:
        return (L, solve1(c2, a2, b - a1 * L, 0, inf))
    elif b == 0:
        return (0, 0)
    else:
        if U != inf:
            U = min(U, b // a1)
        if L != 0 or U != inf:
            xp, yp = solve(c1, c2, a1, a2, b - ((b - a1 * U + a2 - 1) // a2) * a2 - a1 * L, 0, inf)
            xp += L
            yp += (b - a1 * U + a2 - 1) // a2
            return max((U, (b - a1 * U) // a2), (xp, yp), key=value)
        if a1 < a2:
            x, y = solve(c2, c1, a2, a1, b, 0, inf)
            return y, x
        k = a1 // a2
        p = a1 - k * a2
        x, y = solve(c1 - c2 * k, c2, p, a2, b - k * (b // a1) * a2, 0, b // a1)
        y -= k *x
        y += (b // a1) * k
        return x, y

x, y = solve(p, q, r, s, c, 0, inf)
print x * p + q * y
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from bootcamp import Basebootcamp

class Comnomandcandiesbootcamp(Basebootcamp):
    def __init__(self, **params):
        # 参数定义：C是总重量限制，Hr是红色糖果的快乐值，Hb是蓝色糖果的快乐值，
        # Wr是红色糖果的重量，Wb是蓝色糖果的重量
        self.params = {
            'C': params.get('C', 10),
            'Hr': params.get('Hr', 3),
            'Hb': params.get('Hb', 5),
            'Wr': params.get('Wr', 2),
            'Wb': params.get('Wb', 3)
        }
    
    def case_generator(self):
        # 生成随机的参数，考虑较大的范围
        C = random.randint(1, 10**9)
        Hr = random.randint(1, 10**9)
        Hb = random.randint(1, 10**9)
        Wr = random.randint(1, 10**9)
        Wb = random.randint(1, 10**9)
        
        # 计算正确解
        max_joy = 0
        max_red = C // Wr if Wr != 0 else 0
        for red in range(0, max_red + 1):
            weight_left = C - red * Wr
            if weight_left < 0:
                continue
            blue = weight_left // Wb if Wb != 0 else 0
            current_joy = red * Hr + blue * Hb
            if current_joy > max_joy:
                max_joy = current_joy
        
        # 返回问题实例
        return {
            'C': C,
            'Hr': Hr,
            'Hb': Hb,
            'Wr': Wr,
            'Wb': Wb,
            'correct_joy': max_joy
        }
    
    @staticmethod
    def prompt_func(question_case):
        prompt = f"""
        你是Om Nom，你有两种糖果可以选择：红色糖果和蓝色糖果。红色糖果每个重{question_case['Wr']}克，能带来{question_case['Hr']}的快乐值；
        蓝色糖果每个重{question_case['Wb']}克，能带来{question_case['Hb']}的快乐值。你最多可以吃{question_case['C']}克的糖果，
        但必须整颗整颗地吃。你的目标是在不超过重量限制的情况下，获得最大的快乐值。
        
        请计算你最多能获得多少快乐值？把答案放在[answer]标签中，例如：
        [answer]16[/answer]
        """
        return prompt.strip()
    
    @staticmethod
    def extract_output(output):
        # 查找最后一个[answer]标签内的内容
        start = output.rfind('[answer]')
        if start == -1:
            return None
        end = output.find('[/answer]', start)
        if end == -1:
            return None
        answer_str = output[start + 8:end].strip()
        try:
            return int(answer_str)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 比较提取的答案与正确解
        if solution is None:
            return False
        return solution == identity['correct_joy']
