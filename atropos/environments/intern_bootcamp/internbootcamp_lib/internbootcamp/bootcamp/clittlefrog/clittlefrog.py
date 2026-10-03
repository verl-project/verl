"""# 

### 谜题描述
Once upon a time a little frog whose name was Vasya decided to travel around his home swamp. Overall there are n mounds on the swamp, located on one line. The distance between the neighboring mounds is one meter. Vasya wants to visit all the mounds in one day; besides, he wants to visit each one exactly once. For that he makes a route plan, to decide the order in which to jump on the mounds. Vasya can pick any mound as the first one. He thinks it boring to jump two times at the same distance. That's why he wants any two jumps on his route to have different lengths. Help Vasya the Frog and make the plan for him.

Input

The single line contains a number n (1 ≤ n ≤ 104) which is the number of mounds.

Output

Print n integers pi (1 ≤ pi ≤ n) which are the frog's route plan. 

  * All the pi's should be mutually different. 
  * All the |pi–pi + 1|'s should be mutually different (1 ≤ i ≤ n - 1). 



If there are several solutions, output any.

Examples

Input

2


Output

1 2 

Input

3


Output

1 3 2 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
if n==1:
    print 1
    quit()
j=n
i=1
while j!=i-1:
    print i,
    if i!=j:
        print j,
    else:
        quit()
    i+=1
    j-=1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

from bootcamp import Basebootcamp

class Clittlefrogbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)  # 默认n为5

    def case_generator(self):
        # 生成随机n在1到100之间
        n = random.randint(1, 100)
        solution = self._generate_solution(n)
        return {'n': n, 'solution': solution}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        prompt = f"Vasya the frog needs to visit {n} mounds in a line, each exactly once. Each jump between consecutive mounds must be a unique distance. Generate a permutation of 1 to {n} such that the absolute differences between consecutive elements are all unique. Output your answer as space-separated integers within [answer] tags. For example, if n=3, a valid answer is [answer]1 3 2[/answer]."
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.split()))
            return solution
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        # 检查长度是否正确
        if len(solution) != n:
            return False
        # 检查是否是1到n的排列
        if sorted(solution) != list(range(1, n+1)):
            return False
        # 检查相邻差是否互不相同
        diffs = []
        for i in range(n-1):
            diff = abs(solution[i] - solution[i+1])
            if diff in diffs:
                return False
            diffs.append(diff)
        return True

    def _generate_solution(self, n):
        if n == 1:
            return [1]
        solution = []
        i = 1
        j = n
        while i <= j:
            solution.append(i)
            if i != j:
                solution.append(j)
            i += 1
            j -= 1
        # 确保长度正确，处理n为奇数的情况
        return solution[:n]
