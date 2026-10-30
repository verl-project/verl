"""# 

### 谜题描述
Petya loves computer games. Finally a game that he's been waiting for so long came out!

The main character of this game has n different skills, each of which is characterized by an integer ai from 0 to 100. The higher the number ai is, the higher is the i-th skill of the character. The total rating of the character is calculated as the sum of the values ​​of <image> for all i from 1 to n. The expression ⌊ x⌋ denotes the result of rounding the number x down to the nearest integer.

At the beginning of the game Petya got k improvement units as a bonus that he can use to increase the skills of his character and his total rating. One improvement unit can increase any skill of Petya's character by exactly one. For example, if a4 = 46, after using one imporvement unit to this skill, it becomes equal to 47. A hero's skill cannot rise higher more than 100. Thus, it is permissible that some of the units will remain unused.

Your task is to determine the optimal way of using the improvement units so as to maximize the overall rating of the character. It is not necessary to use all the improvement units.

Input

The first line of the input contains two positive integers n and k (1 ≤ n ≤ 105, 0 ≤ k ≤ 107) — the number of skills of the character and the number of units of improvements at Petya's disposal.

The second line of the input contains a sequence of n integers ai (0 ≤ ai ≤ 100), where ai characterizes the level of the i-th skill of the character.

Output

The first line of the output should contain a single non-negative integer — the maximum total rating of the character that Petya can get using k or less improvement units.

Examples

Input

2 4
7 9


Output

2


Input

3 8
17 15 19


Output

5


Input

2 2
99 100


Output

20

Note

In the first test case the optimal strategy is as follows. Petya has to improve the first skill to 10 by spending 3 improvement units, and the second skill to 10, by spending one improvement unit. Thus, Petya spends all his improvement units and the total rating of the character becomes equal to  lfloor frac{100}{10} rfloor + lfloor frac{100}{10} rfloor = 10 + 10 =  20.

In the second test the optimal strategy for Petya is to improve the first skill to 20 (by spending 3 improvement units) and to improve the third skill to 20 (in this case by spending 1 improvement units). Thus, Petya is left with 4 improvement units and he will be able to increase the second skill to 19 (which does not change the overall rating, so Petya does not necessarily have to do it). Therefore, the highest possible total rating in this example is <image>.

In the third test case the optimal strategy for Petya is to increase the first skill to 100 by spending 1 improvement unit. Thereafter, both skills of the character will be equal to 100, so Petya will not be able to spend the remaining improvement unit. So the answer is equal to <image>. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict

n, k = (int(x) for x in raw_input().split())
l = [int(x) for x in raw_input().split()]
counts = [0] * 10
ret = 0
for x in l:
    ret += x / 10
    counts[x%10] += 1
for _ in xrange(9, 0, -1):
    maxCanDo = k / (10 - _)
    actual = min(maxCanDo, counts[_])
    k -= actual * (10 - _)
    ret += actual
print min(10*n, ret + k/10)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
import math

class Cdevelopingskillsbootcamp(Basebootcamp):
    def __init__(self, max_n=1000, max_k=10**6, max_a=100):
        self.max_n = max(max_n, 1)
        self.max_k = max(max_k, 0)
        self.max_a = min(max_a, 100)

    def case_generator(self):
        if random.random() < 0.2:
            return self._generate_edge_case()
        
        n = random.randint(1, self.max_n)
        k = random.randint(0, self.max_k)
        a_list = [random.randint(0, self.max_a) for _ in range(n)]
        
        if random.random() < 0.3 and n > 0:
            a_list[random.randint(0, n-1)] = 100
        
        return {
            'n': n,
            'k': k,
            'a_list': a_list,
            'correct_output': self._calculate_solution(n, k, a_list)
        }

    def _generate_edge_case(self):
        case_type = random.choice([
            'max_skills', 'zero_improvements', 'all_maxed', 
            'large_k', 'minimum_values'
        ])
        
        if case_type == 'max_skills':
            return {
                'n': self.max_n,
                'k': self.max_k,
                'a_list': [100] * self.max_n,
                'correct_output': 10 * self.max_n
            }
        elif case_type == 'zero_improvements':
            a_list = [random.randint(0, 100) for _ in range(random.randint(1, self.max_n))]
            return {
                'n': len(a_list),
                'k': 0,
                'a_list': a_list,
                'correct_output': sum(x//10 for x in a_list)
            }
        elif case_type == 'all_maxed':
            n = random.randint(1, self.max_n)
            return {
                'n': n,
                'k': random.randint(0, self.max_k),
                'a_list': [100]*n,
                'correct_output': 10*n
            }
        elif case_type == 'large_k':
            n = random.randint(1, 100)
            return {
                'n': n,
                'k': 10**7,
                'a_list': [0]*n,
                'correct_output': min(10*n, (sum(0//10 for _ in range(n)) + 10**7//10))
            }
        else:
            return {
                'n': 1,
                'k': 0,
                'a_list': [0],
                'correct_output': 0
            }

    @staticmethod
    def _calculate_solution(n, k, a_list):
        total = sum(x // 10 for x in a_list)
        remainder_counts = [0] * 10  # 索引对应delta值1-9（0位置不使用）

        for x in a_list:
            rem = x % 10
            if rem != 0:
                delta = 10 - rem
                if 1 <= delta <= 9:
                    remainder_counts[delta] += 1

        # 按delta从大到小处理（9到1）
        for delta in range(9, 0, -1):
            if k <= 0:
                break
            count = remainder_counts[delta]
            if count == 0:
                continue

            max_possible = min(k // delta, count)
            total += max_possible
            k -= max_possible * delta

        # 处理剩余k值
        total += k // 10
        return min(total, 10 * n)

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        a_str = ' '.join(map(str, question_case['a_list']))
        prompt = (
            "## 游戏角色技能优化问题\n\n"
            "### 规则说明\n"
            "1. 每个技能值ai（0-100）对应评分⌊ai/10⌋\n"
            "2. 可用k个改进单位（每个+1技能值，不超过100）\n"
            "3. 求最大总评分\n\n"
            f"输入：n={n}, k={k}, 初始值=[{a_str}]\n"
            "输出格式：[answer]答案[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['correct_output']
        except:
            return False
