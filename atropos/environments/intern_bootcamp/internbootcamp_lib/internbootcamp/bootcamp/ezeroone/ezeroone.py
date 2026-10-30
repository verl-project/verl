"""# 

### 谜题描述
Little Petya very much likes playing with little Masha. Recently he has received a game called \"Zero-One\" as a gift from his mother. Petya immediately offered Masha to play the game with him.

Before the very beginning of the game several cards are lain out on a table in one line from the left to the right. Each card contains a digit: 0 or 1. Players move in turns and Masha moves first. During each move a player should remove a card from the table and shift all other cards so as to close the gap left by the removed card. For example, if before somebody's move the cards on the table formed a sequence 01010101, then after the fourth card is removed (the cards are numbered starting from 1), the sequence will look like that: 0100101. 

The game ends when exactly two cards are left on the table. The digits on these cards determine the number in binary notation: the most significant bit is located to the left. Masha's aim is to minimize the number and Petya's aim is to maximize it.

An unpleasant accident occurred before the game started. The kids spilled juice on some of the cards and the digits on the cards got blurred. Each one of the spoiled cards could have either 0 or 1 written on it. Consider all possible variants of initial arrangement of the digits (before the juice spilling). For each variant, let's find which two cards are left by the end of the game, assuming that both Petya and Masha play optimally. An ordered pair of digits written on those two cards is called an outcome. Your task is to find the set of outcomes for all variants of initial digits arrangement.

Input

The first line contains a sequence of characters each of which can either be a \"0\", a \"1\" or a \"?\". This sequence determines the initial arrangement of cards on the table from the left to the right. The characters \"?\" mean that the given card was spoiled before the game. The sequence's length ranges from 2 to 105, inclusive.

Output

Print the set of outcomes for all possible initial digits arrangements. Print each possible outcome on a single line. Each outcome should be represented by two characters: the digits written on the cards that were left by the end of the game. The outcomes should be sorted lexicographically in ascending order (see the first sample).

Examples

Input

????


Output

00
01
10
11


Input

1010


Output

10


Input

1?1


Output

01
11

Note

In the first sample all 16 variants of numbers arrangement are possible. For the variant 0000 the outcome is 00. For the variant 1111 the outcome is 11. For the variant 0011 the outcome is 01. For the variant 1100 the outcome is 10. Regardless of outcomes for all other variants the set which we are looking for will contain all 4 possible outcomes.

In the third sample only 2 variants of numbers arrangement are possible: 111 and 101. For the variant 111 the outcome is 11. For the variant 101 the outcome is 01, because on the first turn Masha can remove the first card from the left after which the game will end.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
ca = '0'
cb = '1'
cu = '?'

def solve(ea, eb, ua, ub, last_char):
  if ea >= eb + 1:
    return ['00']
  if ea + 2 <= eb:
    return ['11']

  if ea == eb or ea + 1 == eb:
    if last_char == ca:
      return ['10']
    elif last_char == cb:
      return ['01']
    elif last_char == cu:
      if ua > 0 and ub > 0:
        return ['10', '01']
      elif ua > 0:
        # last char is necessarily '0'
        return ['10']
      elif ub > 0:
        # last char is necessarily '1'
        return ['01']
      else:
        return []
    else:
      return []

def e(vals):
  a = vals.count(ca)
  b = vals.count(cb)
  u = vals.count(cu)
  last_char = vals[-1]
  results = set()
  #print vals
  #print a, b, u
  for x in xrange(0, u+1):
    ua = x
    ub = u-ua
    ea, eb = a + ua, b + ub
    #print ea, eb, last_char, solve(ea, eb, last_char)
    for val in solve(ea, eb, ua, ub, last_char):
      results.add(val)

  for val in sorted(results):
    print val

#\"\"\"
vals = raw_input()
e(vals)
\"\"\"
e('01????')
print \"---\"
e('1010')
print \"---\"
e('1?1')
print \"---\"
e('101')
print \"---\"
e('00110')
print \"---\"
e('00111')
print \"---\"
e('0011?')
#\"\"\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ezeroonebootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.min_length = params.get('min_length', 2)
        self.max_length = params.get('max_length', 105)
    
    def case_generator(self):
        n = random.randint(self.min_length, self.max_length)
        chars = ['0', '1', '?']
        case = ''.join(random.choices(chars, k=n))
        return {'initial': case}
    
    @staticmethod
    def prompt_func(question_case):
        initial = question_case['initial']
        prompt = f"""
        你是Masha和Petya的游戏分析师，请分析以下卡片序列：{initial}。其中，'?'表示该卡片的数字可以是0或1。游戏规则如下：

        玩家轮流移除卡片，直到剩下两张。Masha先手，她的目标是让最终的两位数尽可能小，而Petya的目标是尽可能大。对于所有可能的初始数字排列，请找出所有可能的最终结果，并按升序排列，每个结果放在单独的一行，格式为两位字符串。将答案放在[answer]标签中。

        例如，输入"1?1"的可能结果为01和11。
        """
        return prompt.strip()
    
    @staticmethod
    def extract_output(output):
        match = re.search(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not match:
            return None
        content = match.group(1).strip()
        lines = content.split('\n')
        results = []
        for line in lines:
            line = line.strip()
            if len(line) == 2 and line.isdigit():
                results.append(line)
        if not results:
            return None
        results = sorted(results)
        return results
    
    @staticmethod
    def solve(ea, eb, ua, ub, last_char):
        if ea >= eb + 1:
            return ['00']
        if ea + 2 <= eb:
            return ['11']
        if ea == eb or ea + 1 == eb:
            if last_char == '0':
                return ['10']
            elif last_char == '1':
                return ['01']
            elif last_char == '?':
                if ua > 0 and ub > 0:
                    return ['10', '01']
                elif ua > 0:
                    return ['10']
                elif ub > 0:
                    return ['01']
                else:
                    return []
            else:
                return []
        return []
    
    @staticmethod
    def compute_possible_outcomes(s):
        if not s:
            return []
        a = s.count('0')
        b = s.count('1')
        u = s.count('?')
        last_char = s[-1]
        results = set()
        for x in range(0, u + 1):
            ea = a + x
            eb = b + (u - x)
            outcome = Ezeroonebootcamp.solve(ea, eb, x, u - x, last_char)
            for o in outcome:
                results.add(o)
        return sorted(results)
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        initial = identity['initial']
        expected = Ezeroonebootcamp.compute_possible_outcomes(initial)
        return solution == expected
