"""# 

### 谜题描述
You are given a permutation p of length n. Remove one element from permutation to make the number of records the maximum possible.

We remind that in a sequence of numbers a1, a2, ..., ak the element ai is a record if for every integer j (1 ≤ j < i) the following holds: aj < ai. 

Input

The first line contains the only integer n (1 ≤ n ≤ 105) — the length of the permutation.

The second line contains n integers p1, p2, ..., pn (1 ≤ pi ≤ n) — the permutation. All the integers are distinct.

Output

Print the only integer — the element that should be removed to make the number of records the maximum possible. If there are multiple such elements, print the smallest one.

Examples

Input

1
1


Output

1


Input

5
5 1 2 3 4


Output

5

Note

In the first example the only element can be removed.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import bisect

n = int(raw_input())
p = map(int, raw_input().split())
# nums_sorted = [p[0]]
nums_sorted = []
record_prevented = [0 for i in range(n+1)]
# record_prevented[p[0]] = -1
for num in p:
    bisect.insort(nums_sorted, num)
    ind = nums_sorted.index(num)
    if ind == len(nums_sorted) - 1:
        record_prevented[num] = -1
    elif ind == len(nums_sorted) - 2:
        record_prevented[nums_sorted[-1]] += 1
    if len(nums_sorted) > 2:
        nums_sorted.pop(0)
mx, mx_num = -1, 1
for num, rec_prev in enumerate(record_prevented):
    if rec_prev > mx and num > 0:
        mx = rec_prev
        mx_num = num
print mx_num
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import bisect
from bootcamp import Basebootcamp

class Cremoveextraonebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10):
        """
        Initialize permutation record puzzle bootcamp with parameters for permutation size.
        
        Args:
            min_n (int): Minimum length of permutation (inclusive)
            max_n (int): Maximum length of permutation (inclusive)
        """
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        """
        Generate a permutation puzzle case with optimal removable element.
        
        Returns:
            dict: {'n', 'p', 'answer'} where p is permutation list, answer is optimal element
        """
        n = random.randint(self.min_n, self.max_n)
        p = list(range(1, n+1))
        random.shuffle(p)
        answer = self._compute_correct_answer(p)
        return {
            'n': n,
            'p': p,
            'answer': answer
        }
    
    @staticmethod
    def _compute_correct_answer(p):
        """
        Compute correct answer using optimized approach from reference code.
        """
        n = len(p)
        nums_sorted = []
        record_prevented = [0] * (n + 2)  # +2 to avoid index issues
        
        for num in p:
            bisect.insort(nums_sorted, num)
            ind = nums_sorted.index(num)
            
            if ind == len(nums_sorted) - 1:
                record_prevented[num] = -1
            elif ind == len(nums_sorted) - 2:
                record_prevented[nums_sorted[-1]] += 1
            
            if len(nums_sorted) > 2:
                nums_sorted.pop(0)
        
        mx, mx_num = -1, float('inf')
        for num in range(1, n+1):
            if record_prevented[num] > mx or (record_prevented[num] == mx and num < mx_num):
                mx = record_prevented[num]
                mx_num = num
        
        return mx_num

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        Format the puzzle case into an instructional prompt with answer format specification.
        """
        n = question_case['n']
        p_str = ' '.join(map(str, question_case['p']))
        return f"""You are participating in a programming competition. Solve the following problem:

**Problem Statement:**
Given a permutation p of length n, remove exactly one element to maximize the number of records in the remaining sequence. A record is an element that is larger than all preceding elements. If multiple elements yield the maximum records, choose the smallest one.

**Input Format:**
- First line: Integer n ({n} in this case)
- Second line: Space-separated permutation elements ({p_str})

**Output Format:**
A single integer - the element to remove. Place your answer between [answer] and [/answer] tags. 

**Example:**
For input:
5
5 1 2 3 4
The correct answer is [answer]5[/answer].

**Your Task:**
Apply the problem-solving process and output your final answer within the specified tags."""

    @staticmethod
    def extract_output(output):
        """
        Extract the last occurrence of an answer between [answer] tags.
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        Verify if extracted solution matches precomputed answer.
        """
        return solution == identity['answer']
