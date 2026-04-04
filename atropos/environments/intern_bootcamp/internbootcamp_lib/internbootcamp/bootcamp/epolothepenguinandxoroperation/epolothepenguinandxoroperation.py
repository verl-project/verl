"""# 

### 谜题描述
Little penguin Polo likes permutations. But most of all he likes permutations of integers from 0 to n, inclusive.

For permutation p = p0, p1, ..., pn, Polo has defined its beauty — number <image>.

Expression <image> means applying the operation of bitwise excluding \"OR\" to numbers x and y. This operation exists in all modern programming languages, for example, in language C++ and Java it is represented as \"^\" and in Pascal — as \"xor\".

Help him find among all permutations of integers from 0 to n the permutation with the maximum beauty.

Input

The single line contains a positive integer n (1 ≤ n ≤ 106).

Output

In the first line print integer m the maximum possible beauty. In the second line print any permutation of integers from 0 to n with the beauty equal to m.

If there are several suitable permutations, you are allowed to print any of them.

Examples

Input

4


Output

20
0 2 1 4 3

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math
def solve(v):
  if (v!=0):
    n=2**int(math.log(v,2)+1)-1
    solve(n-v-1)
    x=v
    while(x>=n-v):
      print x,
      x=x-1
      
u=input()
print u*(u+1)
if (u%2):
  print u,
  solve(u-1)
  print 0,
else:
  print 0,
  solve(u)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

class Epolothepenguinandxoroperationbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100):
        """
        Initialize the bootcamp with parameters for n's range.
        Default minimum n is 1, maximum is 100.
        """
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        """
        Generates a puzzle instance with a random n within the configured range.
        Returns a dictionary containing the generated n.
        """
        n = random.randint(self.min_n, self.max_n)
        return {'n': n}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        Formats the problem into a textual prompt with instructions.
        """
        n = question_case['n']
        max_beauty = n * (n + 1)
        prompt = f"""You are an expert in solving permutation puzzles. Your task is to find the permutation of integers from 0 to {n} that maximizes the sum of each element Epolothepenguinandxoroperation its index. The maximum possible beauty value for this problem is {max_beauty}.

Rules:
1. The permutation must include all integers from 0 to {n}, each exactly once.
2. The output should be two lines: the first line is the maximum beauty value {max_beauty}, and the second line is the permutation separated by spaces.
3. Enclose your answer within [answer] and [/answer] tags.

Example for n=4:
[answer]
20
0 2 1 4 3
[/answer]

Please provide your solution following the above format. Ensure the permutation is correct to achieve the maximum beauty."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        Extracts the permutation from the model's output, expecting the last occurrence within [answer] tags.
        """
        import re
        # Find all answer blocks
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.DOTALL)
        if not answer_blocks:
            return None
        # Take the last answer block
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        # Second line should be the permutation
        permutation_str = lines[1]
        try:
            permutation = list(map(int, permutation_str.split()))
            return permutation
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        Verifies if the extracted permutation is correct.
        """
        if solution is None:
            return False
        n = identity['n']
        # Check if it's a valid permutation of 0 to n
        if sorted(solution) != list(range(n + 1)):
            return False
        # Calculate the beauty value
        total = 0
        for i, num in enumerate(solution):
            total += (num ^ i)
        return total == n * (n + 1)
