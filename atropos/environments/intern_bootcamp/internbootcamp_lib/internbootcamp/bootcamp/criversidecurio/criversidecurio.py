"""# 

### 谜题描述
Arkady decides to observe a river for n consecutive days. The river's water level on each day is equal to some real value.

Arkady goes to the riverside each day and makes a mark on the side of the channel at the height of the water level, but if it coincides with a mark made before, no new mark is created. The water does not wash the marks away. Arkady writes down the number of marks strictly above the water level each day, on the i-th day this value is equal to mi.

Define di as the number of marks strictly under the water level on the i-th day. You are to find out the minimum possible sum of di over all days. There are no marks on the channel before the first day.

Input

The first line contains a single positive integer n (1 ≤ n ≤ 105) — the number of days.

The second line contains n space-separated integers m1, m2, ..., mn (0 ≤ mi < i) — the number of marks strictly above the water on each day.

Output

Output one single integer — the minimum possible sum of the number of marks strictly below the water level among all days.

Examples

Input

6
0 1 0 3 0 2


Output

6


Input

5
0 1 2 1 2


Output

1


Input

5
0 1 1 2 2


Output

0

Note

In the first example, the following figure shows an optimal case.

<image>

Note that on day 3, a new mark should be created because if not, there cannot be 3 marks above water on day 4. The total number of marks underwater is 0 + 0 + 2 + 0 + 3 + 1 = 6.

In the second example, the following figure shows an optimal case.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
nums = list(map(int,raw_input().split()))
cur = 1
ans = 0
arr = [0 for x in range(n)]
for i in range(1,n):
  arr[i] = max(nums[i],arr[i-1])
for i in range(n-1,-1,-1):
  cur = max(cur-1,arr[i])
  ans += cur-nums[i]
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import Dict, Any

from bootcamp import Basebootcamp

class Criversidecuriobootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)  # Default to 5 days if not provided
        # Ensure that 'm' is provided and has the correct length
        self.m = params.get('m', None)
        if self.m is None:
            self.m = [0] * self.n  # Default to all zeros if not provided
        else:
            # Ensure that 'm' has the correct length
            if len(self.m) != self.n:
                raise ValueError("Length of 'm' must be equal to 'n'")
        # Additional parameters can be added if needed

    def case_generator(self) -> Dict[str, Any]:
        n = random.randint(1, 100)  # Random number of days between 1 and 100
        m = []
        m.append(0)  # On day 1, m[0] must be 0
        for i in range(1, n):
            mi = random.randint(0, i-1)  # Valid values for m[i] are 0 <= mi < i
            m.append(mi)
        
        # Now compute the correct answer using the reference solution logic
        arr = [0] * n
        for i in range(1, n):
            arr[i] = max(m[i], arr[i-1])
        cur = 1
        correct_ans = 0
        for i in range(n-1, -1, -1):
            cur = max(cur - 1, arr[i])
            correct_ans += cur - m[i]
        
        return {
            'n': n,
            'm': m,
            'correct_ans': correct_ans
        }
    
    @staticmethod
    def prompt_func(question_case: Dict[str, Any]) -> str:
        n = question_case['n']
        m = question_case['m']
        # Create a string representation of the problem
        prompt = f"Arkady观察了一条河流{n}天。每天他记录了严格高于水位的标记数，分别为：{m}。" \
                f"现在，请你计算每天严格低于水位的标记数的最小总和。" \
                f"请将答案放在[answer]标签中，格式为：[answer]数字[/answer]"
        return prompt
    
    @staticmethod
    def extract_output(output: str) -> int:
        # Extract the last answer in the format [answer]x[/answer]
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        ans_str = matches[-1].strip()
        # Check if the extracted string is a valid integer
        if not ans_str.isdigit():
            return None
        return int(ans_str)
    
    @classmethod
    def _verify_correction(cls, solution: int, identity: Dict[str, Any]) -> bool:
        # Compare the extracted solution with the correct answer
        correct_ans = identity['correct_ans']
        return solution == correct_ans
