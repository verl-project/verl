"""# 

### 谜题描述
A very unusual citizen lives in a far away kingdom — Dwarf Gracula. However, his unusual name is not the weirdest thing (besides, everyone long ago got used to calling him simply Dwarf Greg). What is special about Dwarf Greg — he's been living for over 200 years; besides, he lives in a crypt on an abandoned cemetery and nobody has ever seen him out in daytime. Moreover, nobody has ever seen Greg buy himself any food. That's why nobody got particularly surprised when after the infernal dragon's tragic death cattle continued to disappear from fields. The people in the neighborhood were long sure that the harmless dragon was never responsible for disappearing cattle (considering that the dragon used to be sincere about his vegetarian views). But even that's not the worst part of the whole story.

The worst part is that merely several minutes ago Dwarf Greg in some unintelligible way got inside your house and asked you to help him solve a problem. The point is that a short time ago Greg decided to order a new coffin (knowing his peculiar character, you are not surprised at all). But the problem is: a very long in both directions L-shaped corridor leads to Greg's crypt, and you can't drag just any coffin through that corridor. That's why he asked you to help.

<image>

You've formalized the task on a plane like this: let the corridor's width before and after the turn be equal to a and b correspondingly (see the picture). The corridor turns directly at a right angle, the coffin is a rectangle whose length and width are equal to l and w (l ≥ w) correspondingly. Dwarf Greg has already determined the coffin's length (l), which is based on his height; your task is to determine the coffin's maximally possible width (w), at which it can be brought to the crypt. Besides, due to its large mass (pure marble!) the coffin is equipped with rotating wheels; therefore it is impossible to lift it off the ground, however, arbitrary moves and rotations of the coffin in the plane become possible. The coffin may be rotated arbitrarily just before you drag it into crypt and move through the corridor.

Greg promised that if you help him, he will grant you immortality (I wonder how?). And if you don't, well... trust me, you don't want to know what happens if you don't help him...

Input

The first line contains three space-separated integers a, b and l from the problem's statement (1 ≤ a, b, l ≤ 104).

Output

Print the maximally possible width of a coffin with absolute or relative error no more than 10 - 7. If a coffin with the given length and positive width (the coffin that would meet the conditions from the problem's statement) does not exist, print \"My poor head =(\" (without quotes).

It is guaranteed that if the answer is positive, it will be not less than 10 - 7. All the hacks will also be checked to meet that condition.

Examples

Input

2 2 1


Output

1.0000000


Input

2 2 2


Output

2.0000000

Input

2 2 3


Output

1.3284271


Input

2 2 6


Output

My poor head =(

Note

In the first example the answer is restricted by the coffin's length (remember — coffin's widths should not be larger than it's length).

In the second example it is possible to drag the coffin through the corridor thanks to rotating wheels: firstly, drag it forward by one side while it will not be hampered by the wall, then move it forward by adjacent side perpendicularly to the initial movement direction (remember — arbitrary moves and rotations of the coffin are possible).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
EPS = 1e-8

def cross(a, b):
    return (a[0] * b[1]) - (a[1] * b[0])

def f(a, b, l, x):
    y = (l*l - x*x)**0.5
    return cross( (a-x, b), (-x, y) )

def main():
    a, b, l = map(int, raw_input().split())

    if a > b:
        a, b = b, a

    if l <= a and a <= b:
        print \"%.9lf\" % l
    elif a < l and l <= b:
        print \"%.9lf\" % a
    else:
        lo = 0.0
        hi = float(l)

        while (hi - lo) > EPS:
            x1 = lo + (hi-lo)/3.0
            x2 = lo + (hi-lo)*2.0/3.0

            if f(a, b, l, x1) > f(a, b, l, x2):
                lo = x1
            else:
                hi = x2

        ans = f(a, b, l, lo) / l 

        if ans < EPS:
            print \"My poor head =(\"
        else:
            print \"%.9lf\" % ans

if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
from bootcamp import Basebootcamp

class Ehelpgregthedwarfbootcamp(Basebootcamp):
    def __init__(self, a_max=10**4, b_max=10**4, l_max=10**4):
        self.a_max = a_max
        self.b_max = b_max
        self.l_max = l_max
    
    def case_generator(self):
        a = random.randint(1, self.a_max)
        b = random.randint(1, self.b_max)
        if a > b:
            a, b = b, a
        l = random.randint(1, self.l_max)
        return {
            'a': a,
            'b': b,
            'l': l
        }
    
    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        b = question_case['b']
        l_val = question_case['l']
        prompt = f"""You are a programmer tasked with determining the maximum possible width of a coffin that can be moved through an L-shaped corridor. The corridor has a width of {a} before the turn and {b} after the turn. The coffin's length is {l_val}. Your goal is to find the maximum possible width (w) such that the coffin can be maneuvered through the corridor. The width must be ≤ the length ({l_val}).

Rules:
1. The coffin is a rectangle with length ≥ width.
2. It can be rotated and moved in any direction but must remain in contact with the ground.
3. If no positive width is possible, output "My poor head =(".

Format your answer as a float with at least 7 decimal places (e.g., 1.3284271) or the exact phrase if impossible. Enclose your answer within [answer] and [/answer] tags.

Given a={a}, b={b}, l={l_val}, what is the maximum width w?"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        sol_str = matches[-1].strip()
        if sol_str == "My poor head =(":
            return sol_str
        try:
            return float(sol_str)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        b = identity['b']
        l_val = identity['l']
        
        # Compute correct solution
        if l_val <= a:
            correct = min(l_val, a)
        elif a < l_val <= b:
            correct = a
        else:
            # Ternary search for maximum width
            def compute_f(x):
                y = math.sqrt(l_val**2 - x*x)
                cross = (a - x) * y + x * b  # Correct cross product calculation
                return cross
            
            lo = 0.0
            hi = l_val
            eps = 1e-8
            for _ in range(100):
                if hi - lo < eps:
                    break
                x1 = lo + (hi - lo) / 3
                x2 = hi - (hi - lo) / 3
                f1 = compute_f(x1)
                f2 = compute_f(x2)
                if f1 > f2:
                    hi = x2
                else:
                    lo = x1
            ans = compute_f(lo) / l_val
            if ans < 1e-8:
                correct = "My poor head =("
            else:
                correct = ans
        
        # Validate solution
        if correct == "My poor head =(":
            return solution == correct
        else:
            try:
                sol_num = float(solution)
            except:
                return False
            if sol_num < 0 or sol_num > l_val:
                return False
            abs_err = abs(sol_num - correct)
            rel_err = abs_err / correct if correct != 0 else float('inf')
            return abs_err <= 1e-7 or rel_err <= 1e-7
