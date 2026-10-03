"""# 

### 谜题描述
Dima, Inna and Seryozha have gathered in a room. That's right, someone's got to go. To cheer Seryozha up and inspire him to have a walk, Inna decided to cook something. 

Dima and Seryozha have n fruits in the fridge. Each fruit has two parameters: the taste and the number of calories. Inna decided to make a fruit salad, so she wants to take some fruits from the fridge for it. Inna follows a certain principle as she chooses the fruits: the total taste to the total calories ratio of the chosen fruits must equal k. In other words, <image> , where aj is the taste of the j-th chosen fruit and bj is its calories.

Inna hasn't chosen the fruits yet, she is thinking: what is the maximum taste of the chosen fruits if she strictly follows her principle? Help Inna solve this culinary problem — now the happiness of a young couple is in your hands!

Inna loves Dima very much so she wants to make the salad from at least one fruit.

Input

The first line of the input contains two integers n, k (1 ≤ n ≤ 100, 1 ≤ k ≤ 10). The second line of the input contains n integers a1, a2, ..., an (1 ≤ ai ≤ 100) — the fruits' tastes. The third line of the input contains n integers b1, b2, ..., bn (1 ≤ bi ≤ 100) — the fruits' calories. Fruit number i has taste ai and calories bi.

Output

If there is no way Inna can choose the fruits for the salad, print in the single line number -1. Otherwise, print a single integer — the maximum possible sum of the taste values of the chosen fruits.

Examples

Input

3 2
10 8 1
2 7 1


Output

18


Input

5 3
4 4 4 4 4
2 2 2 2 2


Output

-1

Note

In the first test sample we can get the total taste of the fruits equal to 18 if we choose fruit number 1 and fruit number 2, then the total calories will equal 9. The condition <image> fulfills, that's exactly what Inna wants.

In the second test sample we cannot choose the fruits so as to follow Inna's principle.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
from sys import stdin as cin
from collections import defaultdict

def main():
    n,k = map(int, next(cin).split())
    a = map(int, next(cin).split())
    b = map(int, next(cin).split())
    b = [ x*k for x in b ]

    b2t = defaultdict(lambda:0, { 0 : 0 })
    for t,c in zip(a, b):
        bal = t - c
        b2t2 = b2t.copy()
        for ba, ta in b2t.iteritems():
            b2t2[ba + bal] = max(b2t2[ba + bal], ta + t)
        b2t = b2t2

    return b2t[0] or -1

print main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from collections import defaultdict
import random
from bootcamp import Basebootcamp

def solve(n, k, a_list, b_list):
    b = [x * k for x in b_list]
    b2t = defaultdict(int)
    b2t[0] = 0
    for t, c in zip(a_list, b):
        bal = t - c
        updates = {}
        for ba in list(b2t.keys()):
            new_ba = ba + bal
            new_ta = b2t[ba] + t
            if new_ta > updates.get(new_ba, 0):
                updates[new_ba] = new_ta
        for key, val in updates.items():
            if val > b2t.get(key, 0):
                b2t[key] = val
    max_taste = b2t.get(0, 0)
    return max_taste if max_taste != 0 else -1

class Cdimaandsaladbootcamp(Basebootcamp):
    def __init__(self, max_n=100, k_min=1, k_max=10):
        self.max_n = max_n
        self.k_min = k_min
        self.k_max = k_max
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        k = random.randint(self.k_min, self.k_max)
        
        # Generate valid parameters with guaranteed solution
        valid_case = random.choice([True, False])
        if valid_case:
            # Generate at least one valid fruit
            a = []
            b = []
            # Ensure at least one valid fruit exists
            bi_valid = random.randint(1, 100)
            a.append(k * bi_valid)
            b.append(bi_valid)
            
            # Generate remaining fruits
            for _ in range(n-1):
                if random.random() < 0.3:  # 30% chance to generate valid fruits
                    bi = random.randint(1, 100)
                    ai = k * bi
                else:
                    bi = random.randint(1, 100)
                    ai = random.randint(1, 100)
                a.append(ai)
                b.append(bi)
        else:
            # Generate fruits with all a_i != k*b_i
            while True:
                a = [random.randint(1, 100) for _ in range(n)]
                b = [random.randint(1, 100) for _ in range(n)]
                if not any(a[i] == k * b[i] for i in range(n)):
                    break

        expected_output = solve(n, k, a, b)
        return {
            'n': n,
            'k': k,
            'a': a,
            'b': b,
            'expected_output': expected_output
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        a = ' '.join(map(str, question_case['a']))
        b = ' '.join(map(str, question_case['b']))
        return f"""你需要解决一个水果沙拉组合优化问题：

题目要求：
从n个水果中选择至少一个，使得总味道与总卡路里的比值正好等于k，且总味道最大。如果无解则返回-1。

输入格式：
第一行：两个整数n和k（用空格分隔）
第二行：{n}个整数表示水果的味道值
第三行：{n}个整数表示水果的卡路里

当前测试案例：
{n} {k}
{a}
{b}

请逐步思考并给出最终答案，格式为：[answer]答案[/answer]。例如：[answer]18[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['expected_output']
        except (ValueError, KeyError):
            return False

# 示例验证
if __name__ == "__main__":
    bootcamp = Cdimaandsaladbootcamp()
    case = bootcamp.case_generator()
    print("生成的案例:", case)
    print("\n问题描述:")
    print(bootcamp.prompt_func(case))
