"""# 

### 谜题描述
Do you like summer? Residents of Berland do. They especially love eating ice cream in the hot summer. So this summer day a large queue of n Berland residents lined up in front of the ice cream stall. We know that each of them has a certain amount of berland dollars with them. The residents of Berland are nice people, so each person agrees to swap places with the person right behind him for just 1 dollar. More formally, if person a stands just behind person b, then person a can pay person b 1 dollar, then a and b get swapped. Of course, if person a has zero dollars, he can not swap places with person b.

Residents of Berland are strange people. In particular, they get upset when there is someone with a strictly smaller sum of money in the line in front of them.

Can you help the residents of Berland form such order in the line so that they were all happy? A happy resident is the one who stands first in the line or the one in front of who another resident stands with not less number of dollars. Note that the people of Berland are people of honor and they agree to swap places only in the manner described above.

Input

The first line contains integer n (1 ≤ n ≤ 200 000) — the number of residents who stand in the line.

The second line contains n space-separated integers ai (0 ≤ ai ≤ 109), where ai is the number of Berland dollars of a man standing on the i-th position in the line. The positions are numbered starting from the end of the line. 

Output

If it is impossible to make all the residents happy, print \":(\" without the quotes. Otherwise, print in the single line n space-separated integers, the i-th of them must be equal to the number of money of the person on position i in the new line. If there are multiple answers, print any of them.

Examples

Input

2
11 8


Output

9 10 

Input

5
10 9 7 10 6


Output

:(


Input

3
12 3 3


Output

4 4 10 

Note

In the first sample two residents should swap places, after that the first resident has 10 dollars and he is at the head of the line and the second resident will have 9 coins and he will be at the end of the line. 

In the second sample it is impossible to achieve the desired result.

In the third sample the first person can swap with the second one, then they will have the following numbers of dollars: 4 11 3, then the second person (in the new line) swaps with the third one, and the resulting numbers of dollars will equal to: 4 4 10. In this line everybody will be happy.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
input_list = map(int, raw_input().split())

for i in range(n):
    input_list[i] += i

a = []

input_list.sort()

result = ''

for i in range(1, n):
    if input_list[i - 1] == input_list[i]:
        result = ':('

if result != ':(':
    for i in range(n):
        result += str(input_list[i] - i) + ' '

print(result)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ghappylinebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=1000, solvable_probability=0.5, value_range=(0, 10**9)):
        super().__init__()
        self.min_n = min_n
        self.max_n = max_n
        self.solvable_probability = solvable_probability
        self.value_range = value_range

    def case_generator(self):
        if random.random() < self.solvable_probability:
            return self.generate_solvable_case()
        else:
            return self.generate_unsolvable_case()

    def generate_solvable_case(self):
        n = random.randint(self.min_n, self.max_n)
        current = random.randint(*self.value_range)
        sorted_values = [current]
        for _ in range(1, n):
            current += random.randint(1, 100)
            sorted_values.append(current)
        a = [sorted_values[i] - i for i in range(n)]
        random.shuffle(a)
        return {'n': n, 'a': a}

    def generate_unsolvable_case(self):
        n = random.randint(2, self.max_n)
        # 确保至少有一个重复对
        duplicate_value = random.randint(*self.value_range) + n//2  # 提升冲突概率
        processed_values = [duplicate_value - i for i in range(n)]
        # 随机插入一个重复值
        idx = random.randint(0, n-2)
        processed_values[idx] = processed_values[idx+1] = duplicate_value
        a = [x - i for i, x in enumerate(processed_values)]
        return {'n': n, 'a': a}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        return f"""As the ice cream stall manager in Berland, determine if residents can swap positions (costing $1 per swap) to create a non-decreasing money sequence from front to back.

Input:
{n}
{a}

Rules:
1. Swaps consume $1 from the forward-moving person
2. Positions are numbered from the queue end (position 0 = last)
3. Output the modified amounts or ":(" if impossible

Format your answer within [answer]...[/answer] tags."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            a = identity['a']
            n = identity['n']
            
            # 原题解法逻辑
            processed = [a[i] + i for i in range(n)]
            processed_sorted = sorted(processed)
            
            # 检查是否有相邻重复
            has_duplicate = any(processed_sorted[i] == processed_sorted[i+1] for i in range(n-1))
            
            if has_duplicate:
                return solution.strip() == ":("
            else:
                # 构造正确解
                correct_answer = [processed_sorted[i] - i for i in range(n)]
                user_answer = list(map(int, solution.split()))
                return user_answer == correct_answer
        except Exception:
            return False
