"""# 

### 谜题描述
Vanya has a scales for weighing loads and weights of masses w0, w1, w2, ..., w100 grams where w is some integer not less than 2 (exactly one weight of each nominal value). Vanya wonders whether he can weight an item with mass m using the given weights, if the weights can be put on both pans of the scales. Formally speaking, your task is to determine whether it is possible to place an item of mass m and some weights on the left pan of the scales, and some weights on the right pan of the scales so that the pans of the scales were in balance.

Input

The first line contains two integers w, m (2 ≤ w ≤ 109, 1 ≤ m ≤ 109) — the number defining the masses of the weights and the mass of the item.

Output

Print word 'YES' if the item can be weighted and 'NO' if it cannot.

Examples

Input

3 7


Output

YES


Input

100 99


Output

YES


Input

100 50


Output

NO

Note

Note to the first sample test. One pan can have an item of mass 7 and a weight of mass 3, and the second pan can have two weights of masses 9 and 1, correspondingly. Then 7 + 3 = 9 + 1.

Note to the second sample test. One pan of the scales can have an item of mass 99 and the weight of mass 1, and the second pan can have the weight of mass 100.

Note to the third sample test. It is impossible to measure the weight of the item in the manner described in the input. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

def main():
    W, M = map(int, raw_input().split())
    dig = []


    while(M):
        dig.append(M%W)
        M /= W

    dig.extend([0, 0, 0])


    for i in range(len(dig)):
        if(dig[i] > 1):
            diff = dig[i]-W
           
            if(diff < -1):
                print \"NO\"
                sys.exit()
            
            dig[i+1] += 1
            dig[i] = diff
         

    print \"YES\"
    
##########################################
##########################################
##########################################
if(__name__ == \"__main__\"):
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random

class Cvanyaandscalesbootcamp(Basebootcamp):
    def __init__(self, w_min=2, w_max=10**9, m_min=1, m_max=10**9):
        self.w_min = w_min
        self.w_max = w_max
        self.m_min = m_min
        self.m_max = m_max
    
    def case_generator(self):
        w = random.randint(self.w_min, self.w_max)
        m = random.randint(self.m_min, self.m_max)
        return {'w': w, 'm': m}
    
    @staticmethod
    def prompt_func(question_case):
        w = question_case['w']
        m = question_case['m']
        prompt = f"""Vanya has a balance scale and weights in denominations of w⁰, w¹, ..., w¹⁰⁰ grams (w is an integer ≥2). Determine if you can measure an object of mass {m}g by:
- Placing the object on the left pan
- Placing some weights on either pan
such that the total weight on both pans is equal.

Output 'YES' or 'NO' enclosed in [answer] tags. Example: [answer]YES[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        last_match = matches[-1].strip().upper()
        return last_match if last_match in {'YES', 'NO'} else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == cls.check_balance(identity['w'], identity['m'])
    
    @staticmethod
    def check_balance(w, m):
        if w < 2 or m < 1:
            return "NO"
        
        # Convert m to base-w digits
        current_m = m
        dig = []
        while current_m > 0:
            dig.append(current_m % w)
            current_m //= w
        
        # Process digits with dynamic padding
        dig += [0] * (len(dig) + 2)  # Ensure sufficient padding
        
        for i in range(len(dig)):
            if dig[i] > 1:
                # Dynamically extend array if needed
                while i + 1 >= len(dig):
                    dig.append(0)
                
                diff = dig[i] - w
                if diff < -1:
                    return "NO"
                
                dig[i] = diff
                dig[i+1] += 1
        
        # Final validation check
        return "YES" if all(d in (0, 1) for d in dig) else "NO"
