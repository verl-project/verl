"""# 

### 谜题描述
Marina loves strings of the same length and Vasya loves when there is a third string, different from them in exactly t characters. Help Vasya find at least one such string.

More formally, you are given two strings s1, s2 of length n and number t. Let's denote as f(a, b) the number of characters in which strings a and b are different. Then your task will be to find any string s3 of length n, such that f(s1, s3) = f(s2, s3) = t. If there is no such string, print  - 1.

Input

The first line contains two integers n and t (1 ≤ n ≤ 105, 0 ≤ t ≤ n).

The second line contains string s1 of length n, consisting of lowercase English letters.

The third line contain string s2 of length n, consisting of lowercase English letters.

Output

Print a string of length n, differing from string s1 and from s2 in exactly t characters. Your string should consist only from lowercase English letters. If such string doesn't exist, print -1.

Examples

Input

3 2
abc
xyc


Output

ayd

Input

1 0
c
b


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,t=map(int,raw_input().split())
st1=str(raw_input())
st2=str(raw_input())
c=0
ans=[0]*(n+2)
def cal(a,b):
    if a!=\"a\" and b!=\"a\":
        return \"a\"
    if a!=\"b\" and b!=\"b\":
        return \"b\"
    if a!=\"c\" and b!=\"c\":
        return \"c\"
for i in range(0,n):
    if st1[i]!=st2[i]:
        c=c+1
if 1==2:
    print -1
else:
    ans=[]
    c1=0
    for i in st2:
        ans.append(i)
    for i in range(0,n):
        if ans[i]!=st1[i]:
            c1=c1+1
    t1=c1
    t2=0
    l=0
    for i in range(0,n):
        if st1[i]==st2[i]:
            if t1<t and t2<t :
                t1=t1+1
                t2=t2+1
                ans[i]=cal(st1[i],st2[i])
        else:
            l=l+1
    vl=2*l-(t1-t2)
    #print t1,t2
    if t1>=t and t2<t:

        for i in range(0,n):
            if st1[i]!=st2[i]:
                if t1 > t and t2 < t:
                    ans[i]=st1[i]
                    t1=t1-1
                    t2=t2+1
                    #print ans[i],i
                elif t1<=t and t2<t:
                    ans[i]=cal(st1[i],st2[i])
                    t1=t1
                    t2 = t2+1
                #elif t1==t and t2<t:
                    #ans[i]=

            #print t1,t2
    if t1==t and t2==t:
        print \"\".join(ans)

    else:
        print -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cmarinaandvasyabootcamp(Basebootcamp):
    def __init__(self, max_n=10):
        self.max_n = max_n
    
    def case_generator(self):
        # Decide case type: 70% solvable, 30% impossible
        if random.random() < 0.7:
            if random.choice([True, False]):
                # Case 1: Generate s3 first (two modified variants)
                n = random.randint(1, self.max_n)
                t = random.randint(0, n)
                s3 = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(n)])
                
                s1 = list(s3)
                indices = random.sample(range(n), t)
                for i in indices:
                    available = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c != s1[i]]
                    s1[i] = random.choice(available)
                s1 = ''.join(s1)
                
                s2 = list(s3)
                indices = random.sample(range(n), t)
                for i in indices:
                    available = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c != s2[i]]
                    s2[i] = random.choice(available)
                s2 = ''.join(s2)
                
                return {
                    'n': n, 't': t, 's1': s1, 's2': s2, 'possible': True
                }
            else:
                # Case 2: Generate identical s1/s2 with valid solution
                n = random.randint(1, self.max_n)
                t = random.randint(0, n)
                s1 = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(n)])
                s2 = s1
                return {
                    'n': n, 't': t, 's1': s1, 's2': s2, 'possible': True
                }
        else:
            if random.random() < 0.5:
                # Classic impossible case: n=1, t=0, different strings
                return {
                    'n': 1, 't': 0, 's1': 'a', 's2': 'b', 'possible': False
                }
            else:
                # Generate case with t < minimal required t
                n = random.randint(2, self.max_n)
                d = random.randint(1, n)
                s1 = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(n)])
                s2 = list(s1)
                # Create d differences
                indices = random.sample(range(n), d)
                for i in indices:
                    available = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c != s2[i]]
                    s2[i] = random.choice(available)
                s2 = ''.join(s2)
                # Calculate minimal required t
                t_min = (d + 1) // 2
                t = max(0, t_min - 1)
                return {
                    'n': n, 't': t, 's1': s1, 's2': s2, 'possible': False
                }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        t = question_case['t']
        s1 = question_case['s1']
        s2 = question_case['s2']
        prompt = f"""Given two strings of length {n}, find a third string differing from both in exactly {t} positions.
        
Input:
{n} {t}
{s1}
{s2}

Rules:
1. Output must be length {n} and use lowercase letters
2. If impossible, output -1
3. Format answer within [answer][/answer] tags

Example valid response:
[answer]axyz[/answer] or [answer]-1[/answer]

Now solve this:"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        ans = matches[-1].strip().lower()
        if ans == "-1":
            return -1
        # Validate characters
        if all(c in 'abcdefghijklmnopqrstuvwxyz' for c in ans):
            return ans
        return None  # Invalid characters
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        t = identity['t']
        s1 = identity['s1']
        s2 = identity['s2']
        possible = identity['possible']
        
        if solution == -1:
            return not possible
        if not isinstance(solution, str) or len(solution) != n:
            return False
        # Check character validity
        if not all(c in 'abcdefghijklmnopqrstuvwxyz' for c in solution):
            return False
        # Calculate Hamming distances
        diff_s1 = sum(1 for a, b in zip(s1, solution) if a != b)
        diff_s2 = sum(1 for a, b in zip(s2, solution) if a != b)
        return diff_s1 == t and diff_s2 == t
