"""# 

### 谜题描述
Anton likes to listen to fairy tales, especially when Danik, Anton's best friend, tells them. Right now Danik tells Anton a fairy tale:

\"Once upon a time, there lived an emperor. He was very rich and had much grain. One day he ordered to build a huge barn to put there all his grain. Best builders were building that barn for three days and three nights. But they overlooked and there remained a little hole in the barn, from which every day sparrows came through. Here flew a sparrow, took a grain and flew away...\"

More formally, the following takes place in the fairy tale. At the beginning of the first day the barn with the capacity of n grains was full. Then, every day (starting with the first day) the following happens:

  * m grains are brought to the barn. If m grains doesn't fit to the barn, the barn becomes full and the grains that doesn't fit are brought back (in this problem we can assume that the grains that doesn't fit to the barn are not taken into account). 
  * Sparrows come and eat grain. In the i-th day i sparrows come, that is on the first day one sparrow come, on the second day two sparrows come and so on. Every sparrow eats one grain. If the barn is empty, a sparrow eats nothing. 



Anton is tired of listening how Danik describes every sparrow that eats grain from the barn. Anton doesn't know when the fairy tale ends, so he asked you to determine, by the end of which day the barn will become empty for the first time. Help Anton and write a program that will determine the number of that day!

Input

The only line of the input contains two integers n and m (1 ≤ n, m ≤ 1018) — the capacity of the barn and the number of grains that are brought every day.

Output

Output one integer — the number of the day when the barn will become empty for the first time. Days are numbered starting with one.

Examples

Input

5 2


Output

4


Input

8 1


Output

5

Note

In the first sample the capacity of the barn is five grains and two grains are brought every day. The following happens:

  * At the beginning of the first day grain is brought to the barn. It's full, so nothing happens. 
  * At the end of the first day one sparrow comes and eats one grain, so 5 - 1 = 4 grains remain. 
  * At the beginning of the second day two grains are brought. The barn becomes full and one grain doesn't fit to it. 
  * At the end of the second day two sparrows come. 5 - 2 = 3 grains remain. 
  * At the beginning of the third day two grains are brought. The barn becomes full again. 
  * At the end of the third day three sparrows come and eat grain. 5 - 3 = 2 grains remain. 
  * At the beginning of the fourth day grain is brought again. 2 + 2 = 4 grains remain. 
  * At the end of the fourth day four sparrows come and eat grain. 4 - 4 = 0 grains remain. The barn is empty. 



So the answer is 4, because by the end of the fourth day the barn becomes empty.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
a, b = (int(x) for x in raw_input().split())
if a <= b:
    print a
else:
    ret = b
    lhs = 1
    rhs = int(1e18)
    while lhs < rhs:
        mid = (lhs+rhs)/2
        if mid*(mid+1)/2 >= a-b:
            rhs = mid
        else:
            lhs = mid+1
    print ret + lhs
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cantonandfairytalebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'max_n': params.get('max_n', 10**18),
            'max_m': params.get('max_m', 10**18),
        }
    
    def case_generator(self):
        max_n = self.params['max_n']
        max_m = self.params['max_m']
        n = random.randint(1, max_n)
        m = random.randint(1, max_m)
        return {'n': n, 'm': m}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        prompt = f"""Anton's barn has a capacity of {n} grains. Every morning, exactly {m} grains are added to the barn. If the barn is full, any excess grains are taken back. Each night of the i-th day, i sparrows come and each eat one grain. If the barn is empty, the sparrows eat nothing.

Your task is to determine which day will be the first when the barn becomes empty after the sparrows eat. Please provide the day number as an integer enclosed within [answer] and [/answer]. For example, if the answer is the 4th day, write [answer]4[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _calculate_correct_day(cls, n, m):
        if n <= m:
            return n
        remaining = n - m
        low, high = 1, 10**18
        while low < high:
            mid = (low + high) // 2
            total = mid * (mid + 1) // 2
            if total >= remaining:
                high = mid
            else:
                low = mid + 1
        return m + low
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        try:
            solution_int = int(solution)
        except:
            return False
        correct_day = cls._calculate_correct_day(n, m)
        return solution_int == correct_day
