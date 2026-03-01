"""# 

### 谜题描述
A film festival is coming up in the city N. The festival will last for exactly n days and each day will have a premiere of exactly one film. Each film has a genre — an integer from 1 to k.

On the i-th day the festival will show a movie of genre ai. We know that a movie of each of k genres occurs in the festival programme at least once. In other words, each integer from 1 to k occurs in the sequence a1, a2, ..., an at least once.

Valentine is a movie critic. He wants to watch some movies of the festival and then describe his impressions on his site.

As any creative person, Valentine is very susceptive. After he watched the movie of a certain genre, Valentine forms the mood he preserves until he watches the next movie. If the genre of the next movie is the same, it does not change Valentine's mood. If the genres are different, Valentine's mood changes according to the new genre and Valentine has a stress.

Valentine can't watch all n movies, so he decided to exclude from his to-watch list movies of one of the genres. In other words, Valentine is going to choose exactly one of the k genres and will skip all the movies of this genre. He is sure to visit other movies.

Valentine wants to choose such genre x (1 ≤ x ≤ k), that the total number of after-movie stresses (after all movies of genre x are excluded) were minimum.

Input

The first line of the input contains two integers n and k (2 ≤ k ≤ n ≤ 105), where n is the number of movies and k is the number of genres.

The second line of the input contains a sequence of n positive integers a1, a2, ..., an (1 ≤ ai ≤ k), where ai is the genre of the i-th movie. It is guaranteed that each number from 1 to k occurs at least once in this sequence.

Output

Print a single number — the number of the genre (from 1 to k) of the excluded films. If there are multiple answers, print the genre with the minimum number.

Examples

Input

10 3
1 1 2 3 2 3 3 1 1 3


Output

3

Input

7 3
3 1 3 2 3 1 2


Output

1

Note

In the first sample if we exclude the movies of the 1st genre, the genres 2, 3, 2, 3, 3, 3 remain, that is 3 stresses; if we exclude the movies of the 2nd genre, the genres 1, 1, 3, 3, 3, 1, 1, 3 remain, that is 3 stresses; if we exclude the movies of the 3rd genre the genres 1, 1, 2, 2, 1, 1 remain, that is 2 stresses.

In the second sample whatever genre Valentine excludes, he will have exactly 3 stresses.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
##
# File Name: 250C.py
# Created Time:  2013/1/21 10:12:10
# Author: iyixiang
# Description: 
# Last Modified: 2013-01-21 11:03:21
##

n, k = map(int, raw_input().split())
s = map(int, raw_input().split())
ns = [s[0]]
for i in range(1, n):
    if s[i] != s[i-1]:
        ns.append(s[i])
cnt = [0]*(k+1)
for i, x in enumerate(ns):
    if i >= 1 and ns[i] != ns[i-1]:
        cnt[x] += 1
    if i < len(ns) -1 and ns[i] != ns[i+1]:
        cnt[x] += 1
    if i >= 1 and i < len(ns) -1 and ns[i+1] != ns[i-1]:
        cnt[x] -= 1

print(cnt.index(max(cnt)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from random import shuffle, choices
from bootcamp import Basebootcamp

class Cmoviecriticsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 10)
        self.k = params.get('k', 3)
        if self.k > self.n:
            self.k = self.n  # 确保k不超过n

    def case_generator(self):
        n = self.n
        k = self.k

        # 生成包含每个类型至少一次的电影序列
        a = list(range(1, k + 1))
        remaining = n - k
        if remaining > 0:
            a += choices(range(1, k + 1), k=remaining)
        shuffle(a)  # 打乱顺序以增加随机性

        # 计算每个类型被排除后的压力次数
        def calculate_stress(x):
            filtered = [num for num in a if num != x]
            if len(filtered) <= 1:
                return 0
            stress = 0
            for i in range(1, len(filtered)):
                if filtered[i] != filtered[i-1]:
                    stress += 1
            return stress

        min_stress = float('inf')
        best_x = None
        for x in range(1, k + 1):
            stress = calculate_stress(x)
            if stress < min_stress:
                min_stress = stress
                best_x = x
            elif stress == min_stress:
                if x < best_x:
                    best_x = x

        # 返回问题实例
        return {
            'n': n,
            'k': k,
            'a': a,
            'correct_answer': best_x
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        a = question_case['a']
        a_str = ', '.join(map(str, a))
        prompt = (
            f"电影节有{n}天，每天放一部电影。电影的类型序列是：{a_str}。\n"
            f"Valentine想排除一个类型，使得剩下的电影中的类型变化次数最少。你应该排除哪个类型？\n"
            f"请把答案放在[answer]标签中，例如[answer]3[/answer]。"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        start = output.rfind('[answer]')
        if start == -1:
            return None
        end = output.find('[/answer]', start)
        if end == -1:
            return None
        answer_str = output[start + 8:end]
        try:
            return int(answer_str)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
