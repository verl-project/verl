"""# 

### 谜题描述
The semester is already ending, so Danil made an effort and decided to visit a lesson on harmony analysis to know how does the professor look like, at least. Danil was very bored on this lesson until the teacher gave the group a simple task: find 4 vectors in 4-dimensional space, such that every coordinate of every vector is 1 or  - 1 and any two vectors are orthogonal. Just as a reminder, two vectors in n-dimensional space are considered to be orthogonal if and only if their scalar product is equal to zero, that is: 

<image>.

Danil quickly managed to come up with the solution for this problem and the teacher noticed that the problem can be solved in a more general case for 2k vectors in 2k-dimensinoal space. When Danil came home, he quickly came up with the solution for this problem. Can you cope with it?

Input

The only line of the input contains a single integer k (0 ≤ k ≤ 9).

Output

Print 2k lines consisting of 2k characters each. The j-th character of the i-th line must be equal to ' * ' if the j-th coordinate of the i-th vector is equal to  - 1, and must be equal to ' + ' if it's equal to  + 1. It's guaranteed that the answer always exists.

If there are many correct answers, print any.

Examples

Input

2


Output

++**
+*+*
++++
+**+

Note

Consider all scalar products in example:

  * Vectors 1 and 2: ( + 1)·( + 1) + ( + 1)·( - 1) + ( - 1)·( + 1) + ( - 1)·( - 1) = 0
  * Vectors 1 and 3: ( + 1)·( + 1) + ( + 1)·( + 1) + ( - 1)·( + 1) + ( - 1)·( + 1) = 0
  * Vectors 1 and 4: ( + 1)·( + 1) + ( + 1)·( - 1) + ( - 1)·( - 1) + ( - 1)·( + 1) = 0
  * Vectors 2 and 3: ( + 1)·( + 1) + ( - 1)·( + 1) + ( + 1)·( + 1) + ( - 1)·( + 1) = 0
  * Vectors 2 and 4: ( + 1)·( + 1) + ( - 1)·( - 1) + ( + 1)·( - 1) + ( - 1)·( + 1) = 0
  * Vectors 3 and 4: ( + 1)·( + 1) + ( + 1)·( - 1) + ( + 1)·( - 1) + ( + 1)·( + 1) = 0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import string

def trans(s):
    return s.translate(string.maketrans(\"+*\", \"*+\"))        

def orthogonal(s, n):
    if n == 0:
        print s
    else:
        orthogonal(s+s, n-1)
        orthogonal(s+trans(s), n-1)

k = input()
if k == 0:
    print \"+\"
else:
    orthogonal(\"+\", k)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Charmonyanalysisbootcamp(Basebootcamp):
    def __init__(self, min_k=0, max_k=9):
        if min_k < 0 or max_k > 9:
            raise ValueError("k must be between 0 and 9 inclusive.")
        self.min_k = min_k
        self.max_k = max_k
    
    def case_generator(self):
        k = random.randint(self.min_k, self.max_k)
        return {"k": k}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        k = question_case["k"]
        n = 2 ** k
        example = (
            "++**\n"
            "+*+*\n"
            "++++\n"
            "+**+"
        ) if k == 2 else ("+" if k == 0 else "")
        prompt = f"""You are tasked with solving a mathematical puzzle involving orthogonal vectors in a {n}-dimensional space. 

**Problem Statement:**
Find 2^{k} vectors in a 2^{k}-dimensional space where each coordinate is either +1 or -1, such that every pair of distinct vectors is orthogonal. Two vectors are orthogonal if their dot product equals zero.

**Input Specification:**
- The integer k is {k} (0 ≤ k ≤ 9).

**Output Format:**
- Print 2^{k} lines, each containing 2^{k} characters.
- Use '+' for +1 and '*' for -1.
  
**Example for k=2:**
{example}

**Answer Submission:**
Place your final answer between [answer] and [/answer] tags. Each vector must be on a separate line."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        return lines
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        k = identity["k"]
        n = 2 ** k
        if len(solution) != n:
            return False
        for row in solution:
            if len(row) != n or any(c not in '+*' for c in row):
                return False
        vectors = []
        for row in solution:
            vectors.append([1 if c == '+' else -1 for c in row])
        for i in range(n):
            for j in range(i + 1, n):
                if sum(a * b for a, b in zip(vectors[i], vectors[j])) != 0:
                    return False
        return True
