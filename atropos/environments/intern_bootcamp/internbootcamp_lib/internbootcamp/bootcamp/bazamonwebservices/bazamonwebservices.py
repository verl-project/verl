"""# 

### 谜题描述
Your friend Jeff Zebos has been trying to run his new online company, but it's not going very well. He's not getting a lot of sales on his website which he decided to call Azamon. His big problem, you think, is that he's not ranking high enough on the search engines. If only he could rename his products to have better names than his competitors, then he'll be at the top of the search results and will be a millionaire.

After doing some research, you find out that search engines only sort their results lexicographically. If your friend could rename his products to lexicographically smaller strings than his competitor's, then he'll be at the top of the rankings!

To make your strategy less obvious to his competitors, you decide to swap no more than two letters of the product names.

Please help Jeff to find improved names for his products that are lexicographically smaller than his competitor's!

Given the string s representing Jeff's product name and the string c representing his competitor's product name, find a way to swap at most one pair of characters in s (that is, find two distinct indices i and j and swap s_i and s_j) such that the resulting new name becomes strictly lexicographically smaller than c, or determine that it is impossible.

Note: String a is strictly lexicographically smaller than string b if and only if one of the following holds:

  * a is a proper prefix of b, that is, a is a prefix of b such that a ≠ b; 
  * There exists an integer 1 ≤ i ≤ min{(|a|, |b|)} such that a_i < b_i and a_j = b_j for 1 ≤ j < i. 

Input

The first line of input contains a single integer t (1 ≤ t ≤ 1500) denoting the number of test cases. The next lines contain descriptions of the test cases.

Each test case consists of a single line containing two space-separated strings s and c (2 ≤ |s| ≤ 5000, 1 ≤ |c| ≤ 5000). The strings s and c consists of uppercase English letters.

It is guaranteed that the sum of |s| in the input is at most 5000 and the sum of the |c| in the input is at most 5000.

Output

For each test case, output a single line containing a single string, which is either

  * the new name which is obtained after swapping no more than one pair of characters that is strictly lexicographically smaller than c. In case there are many possible such strings, you can output any of them; 
  * three dashes (the string \"---\" without quotes) if it is impossible. 

Example

Input


3
AZAMON APPLE
AZAMON AAAAAAAAAAALIBABA
APPLE BANANA


Output


AMAZON
---
APPLE

Note

In the first test case, it is possible to swap the second and the fourth letters of the string and the resulting string \"AMAZON\" is lexicographically smaller than \"APPLE\".

It is impossible to improve the product's name in the second test case and satisfy all conditions.

In the third test case, it is possible not to swap a pair of characters. The name \"APPLE\" is lexicographically smaller than \"BANANA\". Note that there are other valid answers, e.g., \"APPEL\". 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def solve(s, t):
    mns = list(s)
    for i in range(len(s)-2,-1,-1): mns[i] = min(mns[i], mns[i + 1])
    for i in range(len(s)):
        if s[i] != mns[i]:
            j = max(j for j, v in enumerate(s[i:], i) if v == mns[i])
            s = s[:i] + s[j] + s[i+1:j] + s[i] + s[j+1:]
            break
    return s if s < t else '---'

total_string = int(input())
string = []
for i in range(total_string): 
    string.append(raw_input())

final_string = []
for i in range(total_string): 
    break_point = string[i].index(' ')
    swaping = False
    done = False
    last = False
    s = string[i][0:break_point]
    c = string[i][break_point+1:] #equal wala check karna bake he
    print(solve(s,c))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from typing import Dict, Optional, Any
from bootcamp import Basebootcamp

def solve(s, t):
    s_list = list(s)
    mns = list(s_list)
    for i in range(len(s_list)-2, -1, -1):
        mns[i] = min(mns[i], mns[i+1])
    
    for i in range(len(s_list)):
        if s_list[i] != mns[i]:
            candidates = [j for j, v in enumerate(s_list[i:], i) if v == mns[i]]
            if candidates:
                j = max(candidates)
                s_list[i], s_list[j] = s_list[j], s_list[i]
                break
    s_opt = ''.join(s_list)
    return s_opt if s_opt < t else '---'

class Bazamonwebservicesbootcamp(Basebootcamp):
    def __init__(self, max_s_length=10, max_c_length=10, max_attempts=100):
        self.max_s_length = max_s_length
        self.max_c_length = max_c_length
        self.max_attempts = max_attempts

    def case_generator(self) -> Dict[str, Any]:
        for _ in range(self.max_attempts):
            s_len = random.randint(2, self.max_s_length)
            s = ''.join(random.choices(string.ascii_uppercase, k=s_len))
            s_opt = solve(s, 'Z'*5000)
            if s_opt == '---':
                continue
            
            c = self.construct_c_greater_than(s_opt)
            if c is None or s_opt >= c:
                continue
                
            return {'s': s, 'c': c}
        return {'s': 'APPLE', 'c': 'BANANA'}  # Fallback case

    def construct_c_greater_than(self, s_opt: str) -> Optional[str]:
        s_trunc = s_opt[:self.max_c_length]
        
        # Try character increment
        for i in range(len(s_trunc)):
            if s_trunc[i] < 'Z':
                new_char = chr(ord(s_trunc[i]) + 1)
                new_c = s_trunc[:i] + new_char + 'A'*(self.max_c_length-i-1)
                return new_c[:self.max_c_length]
        
        # Try length extension
        if len(s_opt) < self.max_c_length:
            return s_opt + 'A'*(self.max_c_length - len(s_opt))
        
        return None

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""Jeff's product: {question_case['s']}
Competitor's product: {question_case['c']}
Swap up to two characters to make Jeff's product strictly smaller than the competitor's. 
Output the modified name or "---" if impossible.
Put your answer between [answer] and [/answer]."""

    @staticmethod
    def extract_output(output: str) -> Optional[str]:
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution: str, identity: Dict) -> bool:
        s, c = identity['s'], identity['c']
        
        if solution == '---':
            return solve(s, c) == '---'
        
        if not (len(solution) == len(s) and solution < c):
            return False
        
        if solution == s:
            return s < c
        
        diff = [i for i, (a, b) in enumerate(zip(s, solution)) if a != b]
        return len(diff) == 2 and s[diff[0]] == solution[diff[1]] and s[diff[1]] == solution[diff[0]]
