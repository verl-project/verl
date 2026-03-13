"""# 

### 谜题描述
Katya studies in a fifth grade. Recently her class studied right triangles and the Pythagorean theorem. It appeared, that there are triples of positive integers such that you can construct a right triangle with segments of lengths corresponding to triple. Such triples are called Pythagorean triples.

For example, triples (3, 4, 5), (5, 12, 13) and (6, 8, 10) are Pythagorean triples.

Here Katya wondered if she can specify the length of some side of right triangle and find any Pythagorean triple corresponding to such length? Note that the side which length is specified can be a cathetus as well as hypotenuse.

Katya had no problems with completing this task. Will you do the same?

Input

The only line of the input contains single integer n (1 ≤ n ≤ 109) — the length of some side of a right triangle.

Output

Print two integers m and k (1 ≤ m, k ≤ 1018), such that n, m and k form a Pythagorean triple, in the only line.

In case if there is no any Pythagorean triple containing integer n, print  - 1 in the only line. If there are many answers, print any of them.

Examples

Input

3


Output

4 5

Input

6


Output

8 10

Input

1


Output

-1

Input

17


Output

144 145

Input

67


Output

2244 2245

Note

<image>

Illustration for the first sample.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
bc = int(raw_input())

if bc % 2 != 0:
	loda = ((bc ** 2) - 1)/2
	pakoda = loda + 1
	if loda == 0 or pakoda == 0:
		print -1
	else:
		print loda, pakoda
else:
	loda = ((bc**2)/4) - 1
	pakoda = loda + 2
	if loda == 0 or pakoda == 0:
		print -1
	else:
		print loda, pakoda
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

class Cpythagoreantriplesbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10**9, allow_hypotenuse=True, **params):
        self.min_n = min_n
        self.max_n = max_n
        self.allow_hypotenuse = allow_hypotenuse
        super().__init__(**params)
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        leg_case = self.generate_leg_case(n)
        if leg_case:
            return {'n': n, 'm': leg_case[0], 'k': leg_case[1]}
        if self.allow_hypotenuse:
            hypotenuse_case = self.generate_hypotenuse_case(n)
            if hypotenuse_case:
                return {'n': n, 'm': hypotenuse_case[0], 'k': hypotenuse_case[1]}
        return {'n': n, 'solution': -1}
    
    @staticmethod
    def generate_leg_case(n):
        if n == 1:
            return None
        if n % 2 == 1:
            m = (n**2 - 1) // 2
            k = m + 1
            if m > 0 and k > 0:
                return (m, k)
            return None
        else:
            m = (n**2 // 4) - 1
            k = m + 2
            if m > 0 and k > 0:
                return (m, k)
            return None
    
    @staticmethod
    def generate_hypotenuse_case(n):
        max_m = int((n**2)**0.5)
        for m in range(1, max_m):
            k_squared = n**2 - m**2
            if k_squared <= 0:
                continue
            k = int(k_squared**0.5)
            if k * k == k_squared and k > 0 and m < n and k < n:
                return (m, k)
        return None
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        solution = question_case.get('solution', None)
        example = "例如，当n=3时，输出4 5，因为3²+4²=5²。当n=6时，输出8 10，因为6²+8²=10²。"
        if solution == -1:
            example = "例如，当n=1时，无解，输出-1。"
        prompt = (
            f"Katya最近在学习毕达哥拉斯定理。她想知道，给定一个整数{n}，是否存在一个毕达哥拉斯三元组，其中n可以是直角边或斜边。"
            f"请找出这样的两个正整数m和k，使得n、m、k构成一个毕达哥拉斯三元组。如果没有这样的解，请输出-1。{example}"
            f"请输出答案，将答案放在[answer]标签中。"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        start_tag = "[answer]"
        end_tag = "[/answer]"
        start = output.rfind(start_tag)
        if start == -1:
            return None
        end = output.find(end_tag, start + len(start_tag))
        if end == -1:
            return None
        answer = output[start + len(start_tag):end].strip()
        if answer == "-1":
            return -1
        try:
            m, k = map(int, answer.split())
            return (m, k)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        if solution == -1:
            return identity.get('solution', None) == -1
        m, k = solution
        # Check if n is a leg
        if n**2 + m**2 == k**2:
            return True
        # Check if n is the hypotenuse
        if m**2 + k**2 == n**2 and cls.allow_hypotenuse:
            return True
        return False
