"""# 

### 谜题描述
As you know, Vova has recently become a new shaman in the city of Ultima Thule. So, he has received the shaman knowledge about the correct bracket sequences. The shamans of Ultima Thule have been using lots of different types of brackets since prehistoric times. A bracket type is a positive integer. The shamans define a correct bracket sequence as follows:

  * An empty sequence is a correct bracket sequence. 
  * If {a1, a2, ..., al} and {b1, b2, ..., bk} are correct bracket sequences, then sequence {a1, a2, ..., al, b1, b2, ..., bk} (their concatenation) also is a correct bracket sequence. 
  * If {a1, a2, ..., al} — is a correct bracket sequence, then sequence <image> also is a correct bracket sequence, where v (v > 0) is an integer. 



For example, sequences {1, 1, - 1, 2, - 2, - 1} and {3, - 3} are correct bracket sequences, and {2, - 3} is not.

Moreover, after Vova became a shaman, he learned the most important correct bracket sequence {x1, x2, ..., xn}, consisting of n integers. As sequence x is the most important, Vova decided to encrypt it just in case.

Encrypting consists of two sequences. The first sequence {p1, p2, ..., pn} contains types of brackets, that is, pi = |xi| (1 ≤ i ≤ n). The second sequence {q1, q2, ..., qt} contains t integers — some positions (possibly, not all of them), which had negative numbers in sequence {x1, x2, ..., xn}.

Unfortunately, Vova forgot the main sequence. But he was lucky enough to keep the encryption: sequences {p1, p2, ..., pn} and {q1, q2, ..., qt}. Help Vova restore sequence x by the encryption. If there are multiple sequences that correspond to the encryption, restore any of them. If there are no such sequences, you should tell so.

Input

The first line of the input contains integer n (1 ≤ n ≤ 106). The second line contains n integers: p1, p2, ..., pn (1 ≤ pi ≤ 109).

The third line contains integer t (0 ≤ t ≤ n), followed by t distinct integers q1, q2, ..., qt (1 ≤ qi ≤ n).

The numbers in each line are separated by spaces.

Output

Print a single string \"NO\" (without the quotes) if Vova is mistaken and a suitable sequence {x1, x2, ..., xn} doesn't exist.

Otherwise, in the first line print \"YES\" (without the quotes) and in the second line print n integers x1, x2, ..., xn (|xi| = pi; xqj < 0). If there are multiple sequences that correspond to the encrypting, you are allowed to print any of them.

Examples

Input

2
1 1
0


Output

YES
1 -1


Input

4
1 1 1 1
1 3


Output

YES
1 1 -1 -1


Input

3
1 1 1
0


Output

NO


Input

4
1 2 2 1
2 3 4


Output

YES
1 2 -2 -1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

n = int(input())
P = [int(x) for x in input().split()]

for i in [int(x) - 1 for x in input().split()][1:]:
    P[i] *= -1

stack = []
for i in reversed(range(n)):
    p = P[i]
    if (not stack) or stack[-1] != p:
        stack.append(abs(p))
        P[i] = -abs(p)
    else:
        stack.pop()

if not stack:
    print 'YES'
    print ' '.join(str(x) for x in P)
else:
    print 'NO'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cmainsequencebootcamp(Basebootcamp):
    def __init__(self, max_n=10, **kwargs):
        super().__init__(**kwargs)
        self.max_n = max_n

    def case_generator(self):
        # 50% chance to generate valid/invalid cases
        if random.choice([True, False]):
            return self._generate_valid_case()
        else:
            return self._generate_invalid_case()

    def _generate_valid_case(self):
        """Generate a valid case with proper bracket sequence"""
        n = random.randint(2, self.max_n // 2) * 2  # Even number
        while True:
            stack = []
            p_list = []
            x_list = []
            q_list = []
            
            # Generate elements in reverse order
            for i in reversed(range(n)):
                if stack:
                    v = stack.pop()
                    x_val = v
                else:
                    v = random.randint(1, 10**3)  # Reduced range for test cases
                    x_val = -v
                    q_list.append(i + 1)  # 1-based position
                    stack.append(v)
                p_list.append(abs(x_val))
                x_list.append(x_val)
            
            # Reverse to get correct order
            p = p_list[::-1]
            x = x_list[::-1]
            q = sorted(q_list)
            
            if self._validate_x(x):
                return {
                    'n': n,
                    'p': p,
                    't': len(q),
                    'q': q
                }

    def _generate_invalid_case(self):
        """Generate invalid case with impossible solution"""
        # Case type 1: Odd length
        if random.choice([True, False]):
            n = random.choice([x for x in range(1, self.max_n+1) if x % 2 != 0])
            p = [random.randint(1, 10**3) for _ in range(n)]
            t = random.randint(0, n)
            q = random.sample(range(1, n+1), t)
        # Case type 2: Valid length but wrong q positions
        else:
            n = random.randint(2, self.max_n // 2) * 2
            p = [random.randint(1, 10**3) for _ in range(n)]
            t = random.randint(0, n)
            q = random.sample(range(1, n+1), t)
            # Ensure at least one q position is invalid
            if q and random.choice([True, False]):
                q[0] = (q[0] % n) + 1  # Modify first position
        
        return {
            'n': n,
            'p': p,
            't': len(q),
            'q': sorted(q)
        }

    @staticmethod
    def _validate_x(x):
        """Validate generated bracket sequence"""
        stack = []
        for num in x:
            if num > 0:
                stack.append(num)
            else:
                if not stack or stack[-1] != -num:
                    return False
                stack.pop()
        return not stack

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""Help Vova recover the correct bracket sequence from:
Input format:
1. First line: n = {question_case['n']}
2. Second line: {' '.join(map(str, question_case['p']))}
3. Third line: {question_case['t']} {' '.join(map(str, question_case['q'])) if question_case['t'] else ''}

Rules:
- Sequence must be valid bracket sequence
- Absolute values must match p sequence
- Negative positions must exactly match q list

Output format:
[answer]
YES/NO
sequence (if YES)
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        # Handle empty solution
        lines = [l.strip() for l in solution.split('\n') if l.strip()]
        if not lines:
            return False

        # Check NO case
        if lines[0].upper() == 'NO':
            # Verify using reference algorithm
            try:
                n = identity['n']
                p = list(identity['p'])
                q_list = list(identity['q'])
                
                # Apply q modifications
                for pos in q_list:
                    if 1 <= pos <= n:
                        p[pos-1] *= -1
                
                # Run reference validation
                stack = []
                for i in reversed(range(n)):
                    val = abs(p[i])
                    if stack and stack[-1] == val:
                        stack.pop()
                    else:
                        stack.append(val)
                        p[i] = -val
                
                return len(stack) != 0
            except:
                return False

        # Check YES case
        if len(lines) < 2 or lines[0].upper() != 'YES':
            return False

        try:
            x = list(map(int, lines[1].split()))
        except:
            return False

        # Basic validation
        if len(x) != identity['n']:
            return False
        
        # Check p matching
        for xi, pi in zip(x, identity['p']):
            if abs(xi) != pi:
                return False
        
        # Check q positions
        q_set = set(identity['q'])
        for i in range(len(x)):
            pos = i + 1
            if pos in q_set:
                if x[i] >= 0:
                    return False
            else:
                if x[i] <= 0:
                    return False
        
        # Validate bracket sequence
        stack = []
        for num in x:
            if num > 0:
                stack.append(num)
            else:
                if not stack or stack[-1] != -num:
                    return False
                stack.pop()
        
        return not stack
