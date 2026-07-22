"""# 

### 谜题描述
Vasya has recently finished writing a book. Now he faces the problem of giving it the title. Vasya wants the title to be vague and mysterious for his book to be noticeable among others. That's why the title should be represented by a single word containing at least once each of the first k Latin letters and not containing any other ones. Also, the title should be a palindrome, that is it should be read similarly from the left to the right and from the right to the left.

Vasya has already composed the approximate variant of the title. You are given the title template s consisting of lowercase Latin letters and question marks. Your task is to replace all the question marks by lowercase Latin letters so that the resulting word satisfies the requirements, described above. Each question mark should be replaced by exactly one letter, it is not allowed to delete characters or add new ones to the template. If there are several suitable titles, choose the first in the alphabetical order, for Vasya's book to appear as early as possible in all the catalogues.

Input

The first line contains an integer k (1 ≤ k ≤ 26) which is the number of allowed alphabet letters. The second line contains s which is the given template. In s only the first k lowercase letters of Latin alphabet and question marks can be present, the length of s is from 1 to 100 characters inclusively.

Output

If there is no solution, print IMPOSSIBLE. Otherwise, a single line should contain the required title, satisfying the given template. The title should be a palindrome and it can only contain the first k letters of the Latin alphabet. At that, each of those k letters must be present at least once. If there are several suitable titles, print the lexicographically minimal one. 

The lexicographical comparison is performed by the standard < operator in modern programming languages. The line a is lexicographically smaller than the line b, if exists such an i (1 ≤ i ≤ |s|), that ai < bi, and for any j (1 ≤ j < i) aj = bj. |s| stands for the length of the given template.

Examples

Input

3
a?c


Output

IMPOSSIBLE


Input

2
a??a


Output

abba


Input

2
?b?a


Output

abba

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
alpha = \"abcdefghijklmnopqrstuvwxyz\"
#print len(alpha)
def getn(s):
  return alpha.find(s)
#print getn('z')
m = input()
line = raw_input()
ok = True
#5print getn(line)
n = len(line)
ret = [' ' for i in range(0, n)]

for i in range(0, n):
  if line[i] == '?':
    ret[i] = line[n - i - 1]
  elif line[n - i - 1] == '?':
    ret[i] = line[i]
  elif line[i] == line[n - i - 1]:
    ret[i] = line[i]
  else:
    ok = False
    break

if not ok:
  print 'IMPOSSIBLE'
else:
  cnt = 0
  used = [False for i in range(0, m)]
  for i in range(0, n):
    if ret[i] == '?':
      cnt += 1
    else:
      used[getn(ret[i])] = True
  unu = 0
  add = ''
  for i in range(0, m):
    if not used[i]:
      unu += 1
      add = add + alpha[i]

  x = cnt / 2
  if cnt % 2 == 1:
    x += 1
  if x < unu:
    print 'IMPOSSIBLE'
  else:
    while len(add) < x:
      add = 'a' + add
    k = 0
    step = 1
    for i in range(0, n):
      if ret[i] == '?':
        #print k, add[k]
        ret[i] = add[k]
        k += step
        if k == x:
          if cnt % 2 == 1:
            k = x - 2
          else:
            k = x - 1
          step = -1

    print ''.join(ret)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

alpha = 'abcdefghijklmnopqrstuvwxyz'

class Ctitlebootcamp(Basebootcamp):
    def __init__(self, min_k=1, max_k=26):
        self.min_k = min_k
        self.max_k = max_k
    
    def case_generator(self):
        k = random.randint(self.min_k, self.max_k)
        n = random.randint(1, 100)
        s = ['?'] * n
        
        # Determine if the case is solvable (avoid impossible cases for k=1)
        is_solvable = random.choice([True, False]) if k > 1 else True
        
        if is_solvable:
            # Generate a valid palindrome template
            palindrome = []
            letters = alpha[:k]
            for i in range((n + 1) // 2):
                char = letters[i % len(letters)] if i < len(letters) else random.choice(letters)
                palindrome.append(char)
            palindrome = palindrome[: (n+1)//2]
            if n % 2 == 0:
                palindrome += palindrome[::-1]
            else:
                palindrome += palindrome[:-1][::-1]
            palindrome = (palindrome * 2)[:n]  # Ensure length
            
            # Replace some characters with '?'
            for i in range(n):
                if random.random() < 0.3:
                    mirror = n - i -1
                    palindrome[i] = '?'
                    if mirror < n and mirror != i:
                        palindrome[mirror] = '?'
            s_template = ''.join(palindrome)
        else:
            # Generate impossible case with conflicting characters
            if k >= 2 and n >= 2:
                i = random.randint(0, (n-1)//2)
                j = n - i -1
                chars = random.sample(alpha[:k], 2)
                s[i] = chars[0]
                s[j] = chars[1]
                s_template = ''.join(s)
            else:
                # Fallback: create a case that can't contain all letters
                k = 3
                s_template = "a??a"
        
        return {'k': k, 's': s_template}
    
    @staticmethod
    def prompt_func(question_case):
        k = question_case['k']
        s = question_case['s']
        allowed = alpha[:k]
        return f"""Vasya's book title must be a palindrome containing each of the first {k} letters ({allowed}). Given template: {s}
Replace '?'s with letters to meet the criteria. Output the lex smallest solution or IMPOSSIBLE.
Put your answer between [answer] and [/answer], e.g.: [answer]abba[/answer]."""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _is_instance_possible(cls, k, s):
        n = len(s)
        ret = list(s)
        valid = True
        for i in range(n):
            j = n - i -1
            if ret[i] == '?' and ret[j] != '?':
                ret[i] = ret[j]
            elif ret[j] == '?' and ret[i] != '?':
                ret[j] = ret[i]
            elif ret[i] != ret[j]:
                valid = False
        if not valid:
            return False
        
        cnt = ret.count('?')
        used = set()
        for c in ret:
            if c != '?':
                used.add(c)
        needed = k - len(used)
        if needed <= 0:
            return True
        return (cnt + 1) // 2 >= needed
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        k = identity['k']
        s = identity['s']
        solution = solution.strip()
        
        # Handle IMPOSSIBLE case
        if solution.upper() == "IMPOSSIBLE":
            return not cls._is_instance_possible(k, s)
        
        # Validate solution
        if solution != solution[::-1]:
            return False
        if len(solution) != len(s):
            return False
        for i, c in enumerate(s):
            if c != '?' and solution[i] != c:
                return False
        if len(set(solution) & set(alpha[:k])) != k:
            return False
        return True
