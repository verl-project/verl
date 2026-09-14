"""# 

### 谜题描述
Palindromic characteristics of string s with length |s| is a sequence of |s| integers, where k-th number is the total number of non-empty substrings of s which are k-palindromes.

A string is 1-palindrome if and only if it reads the same backward as forward.

A string is k-palindrome (k > 1) if and only if: 

  1. Its left half equals to its right half. 
  2. Its left and right halfs are non-empty (k - 1)-palindromes. 



The left half of string t is its prefix of length ⌊|t| / 2⌋, and right half — the suffix of the same length. ⌊|t| / 2⌋ denotes the length of string t divided by 2, rounded down.

Note that each substring is counted as many times as it appears in the string. For example, in the string \"aaa\" the substring \"a\" appears 3 times.

Input

The first line contains the string s (1 ≤ |s| ≤ 5000) consisting of lowercase English letters.

Output

Print |s| integers — palindromic characteristics of string s.

Examples

Input

abba


Output

6 1 0 0 


Input

abacaba


Output

12 4 1 0 0 0 0 

Note

In the first example 1-palindromes are substring «a», «b», «b», «a», «bb», «abba», the substring «bb» is 2-palindrome. There are no 3- and 4-palindromes here.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s=raw_input()
import math
ls=len(s)
ans=[0 for i in range(ls+1)]
dp=[[0 for i in range(ls+1)] for j in range(ls+1)]
for k in range(1,ls+1):
    for i in range(0,ls-k+1):
        j=k+i
        if s[i]!=s[j-1]:
            continue;
        if k==1:
            dp[i][j]=1
            ans[1]+=1
        elif k==2:
            dp[i][j]=2
            ans[2]+=1
        elif dp[i+1][j-1]!=0:
            dp[i][j]=dp[i][int(math.ceil(i+j)/2)]+1
            ans[dp[i][j]]+=1
for i in range(ls-1,0,-1):
    ans[i]+=ans[i+1]
for i in range(1,ls+1):
    print ans[i],
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dpalindromiccharacteristicsbootcamp(Basebootcamp):
    def __init__(self, min_length=1, max_length=6, allowed_chars='abc'):
        super().__init__()
        if min_length < 1 or max_length < min_length:
            raise ValueError("Invalid length parameters")
        self.min_length = min_length
        self.max_length = max_length
        self.allowed_chars = allowed_chars
    
    def case_generator(self):
        length = random.randint(self.min_length, self.max_length)
        s = ''.join(random.choices(self.allowed_chars, k=length))
        expected = self._compute_expected_answer(s)
        return {'string': s, 'expected': expected}
    
    @staticmethod
    def _compute_expected_answer(s):
        n = len(s)
        is_pal = [[False]*(n+1) for _ in range(n+1)]
        max_k = [[0]*(n+1) for _ in range(n+1)]
        
        # Initialize for length 1 and 2
        for i in range(n):
            is_pal[i][i+1] = True
            max_k[i][i+1] = 1
        
        for i in range(n-1):
            j = i+2
            if s[i] == s[j-1]:
                is_pal[i][j] = True
                max_k[i][j] = 2
        
        # Process longer substrings
        for l in range(3, n+1):
            for i in range(n-l+1):
                j = i + l
                if s[i] == s[j-1] and is_pal[i+1][j-1]:
                    is_pal[i][j] = True
                    mid = (i + j) // 2
                    left = max_k[i][mid]
                    right = max_k[mid][j]
                    if left == 0 or right == 0:
                        max_k[i][j] = 1
                    else:
                        max_k[i][j] = min(left, right) + 1
        
        # Collect results
        ans = [0]*(n+2)
        for i in range(n):
            for j in range(i+1, n+1):
                k_val = max_k[i][j]
                if k_val > 0:
                    ans[k_val] += 1
        
        # Accumulate counts
        for k in range(n-1, 0, -1):
            ans[k] += ans[k+1]
        
        return ans[1:n+1]
    
    @staticmethod
    def prompt_func(case) -> str:
        s = case['string']
        return f"""给定字符串{s}，计算其回文特征数组。答案需用[answer]包裹，如[answer]6 1 0 0[/answer]"""
    
    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers: return None
        try: return list(map(int, answers[-1].strip().split()))
        except: return None
    
    @classmethod
    def _verify_correction(cls, solution, case):
        return solution == case['expected']
