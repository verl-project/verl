"""# 

### 谜题描述
Developers often face with regular expression patterns. A pattern is usually defined as a string consisting of characters and metacharacters that sets the rules for your search. These patterns are most often used to check whether a particular string meets the certain rules.

In this task, a pattern will be a string consisting of small English letters and question marks ('?'). The question mark in the pattern is a metacharacter that denotes an arbitrary small letter of the English alphabet. We will assume that a string matches the pattern if we can transform the string into the pattern by replacing the question marks by the appropriate characters. For example, string aba matches patterns: ???, ??a, a?a, aba.

Programmers that work for the R1 company love puzzling each other (and themselves) with riddles. One of them is as follows: you are given n patterns of the same length, you need to find a pattern that contains as few question marks as possible, and intersects with each of the given patterns. Two patterns intersect if there is a string that matches both the first and the second pattern. Can you solve this riddle?

Input

The first line contains a single integer n (1 ≤ n ≤ 105) — the number of patterns. Next n lines contain the patterns.

It is guaranteed that the patterns can only consist of small English letters and symbols '?'. All patterns are non-empty and have the same length. The total length of all the patterns does not exceed 105 characters.

Output

In a single line print the answer to the problem — the pattern with the minimal number of signs '?', which intersects with each of the given ones. If there are several answers, print any of them.

Examples

Input

2
?ab
??b


Output

xab


Input

2
a
b


Output

?


Input

1
?a?b


Output

cacb

Note

Consider the first example. Pattern xab intersects with each of the given patterns. Pattern ??? also intersects with each of the given patterns, but it contains more question signs, hence it is not an optimal answer. Clearly, xab is the optimal answer, because it doesn't contain any question sign. There are a lot of other optimal answers, for example: aab, bab, cab, dab and so on.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys


def main():
    n = int(sys.stdin.readline())
    t = sys.stdin.readlines()
    l = len(t[0])
    result = ''
    if n == 1:
        result = t[0].replace('?', 'x')
    else:
        for y in range(0, l - 1):
            let = t[0][y]
            for x in range(1, n):
                if let == '?' and t[x][y] != '?':
                    let = t[x][y]
                if t[x][y] != let and t[x][y] != '?':
                    result += '?'
                    break
                elif x == n - 1:
                    if let == '?':
                        result += 'x'
                    else:
                        result += let
    print result


main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cpatternbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.params.setdefault('max_n', 5)
        self.params.setdefault('max_length', 5)
        self.params.setdefault('question_prob', 0.3)
    
    def case_generator(self):
        max_length = self.params['max_length']
        max_n = self.params['max_n']
        q_prob = self.params['question_prob']
        
        m = random.randint(1, max_length)
        n = random.randint(1, max_n)
        
        # 生成确定解并保证至少存在一个非空字符位
        while True:
            solution = [random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(m)]
            if any(solution):  # 确保至少一个非空字符用于验证
                break
        
        patterns = []
        for _ in range(n):
            pattern = []
            for c in solution:
                if random.random() < q_prob:
                    pattern.append('?')
                else:
                    pattern.append(c)
            patterns.append(''.join(pattern))
        
        return {'n': n, 'patterns': patterns}
    
    @staticmethod
    def prompt_func(question_case):
        patterns = question_case['patterns']
        problem = f"""You are in a regex optimization competition. Given {len(patterns)} patterns of the same length, find the pattern with minimal '?' that intersects with all given patterns.

Rules:
1. A string matches a pattern if it can be formed by replacing '?'s with any lowercase letter
2. Two patterns intersect if they share at least one common matching string
3. Your answer must have the minimum possible number of '?' characters

Input Patterns:
""" + "\n".join(patterns) + """

Output your final answer within [answer] and [/answer] tags. Example: [answer]a?c[/answer]"""

        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        patterns = identity['patterns']
        m = len(patterns[0]) if patterns else 0
        
        # Basic format validation
        if len(solution) != m or any(c not in 'abcdefghijklmnopqrstuvwxyz?' for c in solution):
            return False
        
        # Validate intersection with all patterns
        for pattern in patterns:
            if not cls._check_intersection(solution, pattern):
                return False
        
        # Validate optimality (minimum question marks)
        for i in range(m):
            if solution[i] == '?':
                possible_chars = set()
                for p in patterns:
                    c = p[i]
                    if c != '?':
                        possible_chars.add(c)
                
                # Case 1: All patterns have '?' in this position
                if not possible_chars:
                    return False  # Should use concrete character
                
                # Case 2: Check if exists valid concrete character
                has_valid_char = False
                for c in possible_chars:
                    if all(p[i] in ('?', c) for p in patterns):
                        has_valid_char = True
                        break
                if has_valid_char:
                    return False  # Could use concrete character but used '?'
        
        return True
    
    @staticmethod
    def _check_intersection(a, b):
        """Check if two patterns can match the same string"""
        return all(ac == '?' or bc == '?' or ac == bc for ac, bc in zip(a, b))
