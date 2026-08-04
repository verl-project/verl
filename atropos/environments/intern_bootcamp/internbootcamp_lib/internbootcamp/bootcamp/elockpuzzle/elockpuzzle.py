"""# 

### 谜题描述
Welcome to another task about breaking the code lock! Explorers Whitfield and Martin came across an unusual safe, inside of which, according to rumors, there are untold riches, among which one can find the solution of the problem of discrete logarithm!

Of course, there is a code lock is installed on the safe. The lock has a screen that displays a string of n lowercase Latin letters. Initially, the screen displays string s. Whitfield and Martin found out that the safe will open when string t will be displayed on the screen.

The string on the screen can be changed using the operation «shift x». In order to apply this operation, explorers choose an integer x from 0 to n inclusive. After that, the current string p = αβ changes to βRα, where the length of β is x, and the length of α is n - x. In other words, the suffix of the length x of string p is reversed and moved to the beginning of the string. For example, after the operation «shift 4» the string «abcacb» will be changed with string «bcacab », since α = ab, β = cacb, βR = bcac.

Explorers are afraid that if they apply too many operations «shift», the lock will be locked forever. They ask you to find a way to get the string t on the screen, using no more than 6100 operations.

Input

The first line contains an integer n, the length of the strings s and t (1 ≤ n ≤ 2 000).

After that, there are two strings s and t, consisting of n lowercase Latin letters each.

Output

If it is impossible to get string t from string s using no more than 6100 operations «shift», print a single number  - 1.

Otherwise, in the first line output the number of operations k (0 ≤ k ≤ 6100). In the next line output k numbers xi corresponding to the operations «shift xi» (0 ≤ xi ≤ n) in the order in which they should be applied.

Examples

Input

6
abacbb
babcba


Output

4
6 3 2 3


Input

3
aba
bba


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def transform(text,i):
    if i == 0:
        return text
    return text[:-i-1:-1] + text[:-i]

def findBeforeCharFromBack(text,char):
    i = 1
    while text[-i] != char:
        i += 1
    return i

def findBeforeCharFromFront(text,char,n):
    i = n
    while text[-i] != char:
        i -= 1
    return i

def solve(text,goal,n):
    realgoal = goal
    solved = 0
    path = []
    if n % 2 == 1:
        if text[-1] != goal[n/2]:
            i = findBeforeCharFromBack(text,goal[n/2])
            path.append(i-1)
            text = transform(text,i-1)
        solved += 1
    while solved < n:
        shift = findBeforeCharFromFront(text,goal[n/2-(solved+2)/2],n)-1
        text = transform(text,shift)
     
        if shift != 0:
            path.append(shift)
        text = transform(text,n)
       
        path.append(n)
        text = transform(text,solved)

        if solved != 0:
            path.append(solved)
        shift = findBeforeCharFromBack(text,goal[(n-1)/2 + (solved+2)/2])
        text = transform(text,shift)
  
        if shift != 0:
            path.append(shift)
        text = transform(text,n-shift-solved-1)
   
        if n-shift-solved-1 != 0:
            path.append(n-shift-solved-1)
        solved += 2
        goal = goal[::-1]
    if goal != realgoal:
        path.append(n)
        text = transform(text,n)
        goal = realgoal
    shiftleft = 0
    while text[shiftleft:] + text[:shiftleft+1] != goal:
        shiftleft += 1
    if shiftleft != 0:
        path.append(n-shiftleft)
        path.append(shiftleft)
        path.append(n)
        text = transform(text,n-shiftleft)
        text = transform(text,shiftleft)
        text = transform(text,n)
    print len(path)
    
    for move in path:
        print move,
        
    

n= input()
text = raw_input()
goal = raw_input()
for char in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','y','x','z']:
    if text.count(char) != goal.count(char):
        print '-1'
        break
else:
    solve(text,goal,n)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
from bootcamp import Basebootcamp

class Elockpuzzlebootcamp(Basebootcamp):
    def __init__(self, n=6, allow_unsolvable=True):
        self.n = n
        self.allow_unsolvable = allow_unsolvable
    
    def case_generator(self):
        # Randomly decide to generate solvable or unsolvable case
        generate_unsolvable = self.allow_unsolvable and random.choice([False, True])
        
        if generate_unsolvable:
            # Generate s and t with different character frequencies
            while True:
                s = ''.join(random.choices(string.ascii_lowercase, k=self.n))
                t = ''.join(random.choices(string.ascii_lowercase, k=self.n))
                if sorted(s) != sorted(t):
                    break
            return {'n': self.n, 's': s, 't': t}
        else:
            # Generate solvable case by applying random shifts
            s = ''.join(random.choices(string.ascii_lowercase, k=self.n))
            current = s
            k = random.randint(0, min(100, 6100))  # Generate up to 100 shifts
            for _ in range(k):
                x = random.randint(0, self.n)
                current = self._apply_shift(current, x, self.n)
            return {'n': self.n, 's': s, 't': current}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        s = question_case['s']
        t = question_case['t']
        prompt = f"""You are an explorer trying to crack a code lock. The lock displays a string of {n} lowercase letters. Initially, it shows "{s}". Your goal is to transform it into "{t}" using "shift x" operations. Each operation chooses x (0 ≤ x ≤ {n}), reverses the last x characters, and moves them to the front.

For example, applying shift 4 to "abcacb" results in "bcacab" (split into "ab" + "cacb", reversed to "bcac" + "ab").

Your task is to find a sequence of up to 6100 shifts to achieve this transformation. If impossible, output -1.

Format your answer within [answer] tags:
- If possible: first line is k (number of operations), second line is x1 x2 ... xk.
- If impossible: single line -1.

Example:
[answer]
4
6 3 2 3
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if not lines:
            return None
        if lines[0] == '-1':
            return -1
        
        try:
            k = int(lines[0])
            if len(lines) < 2:
                return None
            x_list = list(map(int, lines[1].split()))
            if len(x_list) != k:
                return None
            return x_list
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        s = identity['s']
        t = identity['t']
        
        # Check character composition
        if sorted(s) != sorted(t):
            return solution == -1
        
        # Check if solution is valid
        if solution == -1:
            return False
        
        if not isinstance(solution, list) or len(solution) > 6100:
            return False
        
        for x in solution:
            if not (0 <= x <= n):
                return False
        
        # Apply shifts to s
        current = s
        for x in solution:
            current = cls._apply_shift(current, x, n)
            if current is None:
                return False
        
        return current == t
    
    @classmethod
    def _apply_shift(cls, s, x, n):
        if x < 0 or x > n or len(s) != n:
            return None
        if x == 0:
            return s
        return s[-x:][::-1] + s[:-x]
