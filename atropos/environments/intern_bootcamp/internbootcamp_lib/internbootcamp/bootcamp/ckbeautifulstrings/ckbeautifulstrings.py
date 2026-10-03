"""# 

### 谜题描述
You are given a string s consisting of lowercase English letters and a number k. Let's call a string consisting of lowercase English letters beautiful if the number of occurrences of each letter in that string is divisible by k. You are asked to find the lexicographically smallest beautiful string of length n, which is lexicographically greater or equal to string s. If such a string does not exist, output -1.

A string a is lexicographically smaller than a string b if and only if one of the following holds: 

  * a is a prefix of b, but a ≠ b; 
  * in the first position where a and b differ, the string a has a letter that appears earlier in the alphabet than the corresponding letter in b. 

Input

The first line contains a single integer T (1 ≤ T ≤ 10 000) — the number of test cases.

The next 2 ⋅ T lines contain the description of test cases. The description of each test case consists of two lines.

The first line of the description contains two integers n and k (1 ≤ k ≤ n ≤ 10^5) — the length of string s and number k respectively.

The second line contains string s consisting of lowercase English letters.

It is guaranteed that the sum of n over all test cases does not exceed 10^5.

Output

For each test case output in a separate line lexicographically smallest beautiful string of length n, which is greater or equal to string s, or -1 if such a string does not exist.

Example

Input


4
4 2
abcd
3 1
abc
4 3
aaaa
9 3
abaabaaaa


Output


acac
abc
-1
abaabaaab

Note

In the first test case \"acac\" is greater than or equal to s, and each letter appears 2 or 0 times in it, so it is beautiful.

In the second test case each letter appears 0 or 1 times in s, so s itself is the answer.

We can show that there is no suitable string in the third test case.

In the fourth test case each letter appears 0, 3, or 6 times in \"abaabaaab\". All these integers are divisible by 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import threading
import sys

threading.stack_size(16*2048*2048)
sys.setrecursionlimit(100010)


def getnext(index,fre,k,s,flag):
    if sum(fre)>len(s)-index:
        return \"ERROR\"


#    print(index,fre,pre,flag)
    if index==len(s): return \"\"
    cur = ord(s[index])-97

    if not flag:
        nexts = \"\"
        spare = (len(s)-index-sum(fre))
        if spare%k==0:
            nexts += 'a'*(spare//k*k)
        for j in range(26):
            if fre[j]>0:  nexts += chr(j+97)*fre[j]
        return nexts

        
                    



    nexts = \"ERROR\"
    for j in range(cur,26):
        if j>cur and flag: flag = False
        fre[j] -= 1
        if fre[j]<0: fre[j]+=k
        temp = getnext(index+1,fre,k,s,flag)
        if temp!=\"ERROR\": 
            nexts = chr(j+97)+temp
            return nexts
        fre[j] += 1
        if fre[j]==k:  fre[j] = 0


   #print(index,fre,nexts)
            
           
    return nexts




def main():

    T = int(raw_input())
    t = 1

    while t<=T:
        n,k = map(int,raw_input().split())
        s = raw_input()

        

        if n%k>0:
            print -1
            t += 1
            continue

        fre = [0]*26
        ans = getnext(0,fre,k,s,True)

        print ans
        t += 1



threading.Thread(target=main).start()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import sys
import re
import random
from bootcamp import Basebootcamp

sys.setrecursionlimit(1 << 25)

def getnext(index, fre, k, s, flag):
    if sum(fre) > len(s) - index:
        return "ERROR"
    if index == len(s):
        return ""
    cur = ord(s[index]) - 97 if index < len(s) else 0
    if not flag:
        spare = len(s) - index - sum(fre)
        nexts = ""
        if spare % k == 0:
            nexts += 'a' * (spare // k * k)
        for j in range(26):
            if fre[j] > 0:
                nexts += chr(j + 97) * fre[j]
        return nexts
    nexts = "ERROR"
    for j in range(cur, 26):
        new_flag = flag
        if j > cur:
            new_flag = False
        original_j = fre[j]
        fre[j] -= 1
        if fre[j] < 0:
            fre[j] += k
        temp = getnext(index + 1, fre, k, s, new_flag)
        if temp != "ERROR":
            nexts = chr(j + 97) + temp
            fre[j] = original_j
            return nexts
        fre[j] = original_j
    return nexts

def solve(n, k, s):
    if n % k != 0:
        return "-1"
    fre = [0] * 26
    ans = getnext(0, fre, k, s, True)
    return ans if ans != "ERROR" else "-1"

class Ckbeautifulstringsbootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_k=20):
        self.max_n = max_n
        self.max_k = max_k
    
    def case_generator(self):
        while True:
            if random.choice([True, False]):
                # Generate unsolvable case
                while True:
                    n = random.randint(1, self.max_n)
                    k = random.randint(1, self.max_n)
                    if n % k != 0:
                        break
                s = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=n))
                return {'n': n, 'k': k, 's': s}
            else:
                # Generate solvable case
                k = random.randint(1, self.max_k)
                max_multiplier = self.max_n // k
                if max_multiplier < 1:
                    continue
                multiplier = random.randint(1, max_multiplier)
                n = k * multiplier
                s = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=n))
                solution = solve(n, k, s)
                if solution != "-1":
                    return {'n': n, 'k': k, 's': s}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        s = question_case['s']
        prompt = f"""You are given a string s of length {n} consisting of lowercase English letters and an integer k={k}. A string is called beautiful if the count of each character is divisible by k. Your task is to find the lexicographically smallest beautiful string of length {n} that is greater than or equal to s. If no such string exists, output -1.

Input:
- The string s is "{s}"
- The values are n={n}, k={k}

Please provide your answer enclosed within [answer] and [/answer]. For example: [answer]your_answer[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        k = identity['k']
        s = identity['s']
        expected = solve(n, k, s)
        return solution == expected
