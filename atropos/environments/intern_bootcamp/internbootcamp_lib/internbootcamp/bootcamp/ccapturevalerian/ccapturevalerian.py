"""# 

### 谜题描述
It's now 260 AD. Shapur, being extremely smart, became the King of Persia. He is now called Shapur, His majesty King of kings of Iran and Aniran.

Recently the Romans declared war on Persia. They dreamed to occupy Armenia. In the recent war, the Romans were badly defeated. Now their senior army general, Philip is captured by Shapur and Shapur is now going to capture Valerian, the Roman emperor.

Being defeated, the cowardly Valerian hid in a room at the top of one of his castles. To capture him, Shapur has to open many doors. Fortunately Valerian was too scared to make impenetrable locks for the doors.

Each door has 4 parts. The first part is an integer number a. The second part is either an integer number b or some really odd sign which looks like R. The third one is an integer c and the fourth part is empty! As if it was laid for writing something. Being extremely gifted, after opening the first few doors, Shapur found out the secret behind the locks.

c is an integer written in base a, to open the door we should write it in base b. The only bad news is that this R is some sort of special numbering system that is used only in Roman empire, so opening the doors is not just a piece of cake!

Here's an explanation of this really weird number system that even doesn't have zero:

Roman numerals are based on seven symbols: a stroke (identified with the letter I) for a unit, a chevron (identified with the letter V) for a five, a cross-stroke (identified with the letter X) for a ten, a C (identified as an abbreviation of Centum) for a hundred, etc.:

  * I=1
  * V=5
  * X=10
  * L=50
  * C=100
  * D=500
  * M=1000



Symbols are iterated to produce multiples of the decimal (1, 10, 100, 1, 000) values, with V, L, D substituted for a multiple of five, and the iteration continuing: I 1, II 2, III 3, V 5, VI 6, VII 7, etc., and the same for other bases: X 10, XX 20, XXX 30, L 50, LXXX 80; CC 200, DCC 700, etc. At the fourth and ninth iteration, a subtractive principle must be employed, with the base placed before the higher base: IV 4, IX 9, XL 40, XC 90, CD 400, CM 900.

Also in bases greater than 10 we use A for 10, B for 11, etc.

Help Shapur capture Valerian and bring peace back to Persia, especially Armenia.

Input

The first line contains two integers a and b (2 ≤ a, b ≤ 25). Only b may be replaced by an R which indicates Roman numbering system.

The next line contains a single non-negative integer c in base a which may contain leading zeros but its length doesn't exceed 103. 

It is guaranteed that if we have Roman numerals included the number would be less than or equal to 300010 and it won't be 0. In any other case the number won't be greater than 101510.

Output

Write a single line that contains integer c in base b. You must omit leading zeros.

Examples

Input

10 2
1


Output

1


Input

16 R
5


Output

V


Input

5 R
4


Output

IV


Input

2 2
1111001


Output

1111001


Input

12 13
A


Output

A

Note

You can find more information about roman numerals here: http://en.wikipedia.org/wiki/Roman_numerals

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from string import digits
L=\"IVXLCDM__\"
Q=[\"\",\"0\",\"00\",\"000\",\"01\",\"1\",\"10\",\"100\",\"1000\",\"02\"]
def val(x):
	return ord(x)-ord('0') if x in digits else ord(x)-ord('A')+10
def rval(x):
	return chr(ord('A')+x-10) if x>9 else chr(ord('0')+x)
def cal(x,s):
	return Q[x].replace('0',L[s]).replace('1',L[s+1]).replace('2',L[s+2])
a,b=raw_input().split()
a=int(a)
c=raw_input()
r=0
for i in c:
	r=r*a+val(i)
ans=\"\"
if b=='R':
	q=3
	while q>=0:
		ans+=cal(r/(10**q),2*q)
		r%=10**q
		q-=1
	print ans
else:
	b=int(b)
	while r>0:
		ans+=rval(r%b)
		r=r/b
	if ans==\"\":
		ans=\"0\"
	print ans[::-1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from string import digits
from bootcamp import Basebootcamp

class Ccapturevalerianbootcamp(Basebootcamp):
    def __init__(self, roman_prob=0.5, min_a=2, max_a=25, min_b=2, max_b=25):
        self.roman_prob = roman_prob
        self.min_a = min_a
        self.max_a = max_a
        self.min_b = min_b
        self.max_b = max_b

    def case_generator(self):
        is_roman = random.random() < self.roman_prob
        if is_roman:
            a = random.randint(self.min_a, self.max_a)
            b = 'R'
            num = random.randint(1, 3999)
        else:
            a = random.randint(self.min_a, self.max_a)
            b = random.randint(self.min_b, self.max_b)
            num = random.randint(0, 101510)
        
        # 生成基数转换并添加前导零
        c = self.decimal_to_base(num, a)
        max_leading = 1000 - len(c)  # 题目限制长度为10^3
        leading_zeros = random.randint(0, max_leading) if max_leading > 0 else 0
        c = '0' * leading_zeros + c

        correct_answer = self.decimal_to_roman(num) if is_roman else self.decimal_to_base(num, b)
        return {
            'a': a,
            'b': b,
            'c': c,
            'correct_answer': correct_answer
        }

    @staticmethod
    def decimal_to_base(n, base):
        if n == 0:
            return '0'
        digits = []
        while n > 0:
            remainder = n % base
            digits.append(str(remainder) if remainder < 10 else chr(ord('A') + remainder - 10))
            n = n // base
        return ''.join(reversed(digits)) if digits else '0'

    @staticmethod
    def decimal_to_roman(num):
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4, 1
        ]
        syms = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
        ]
        roman_num = []
        i = 0
        while num > 0:
            count = num // val[i]
            roman_num.append(syms[i] * count)
            num -= val[i] * count
            i += 1
        return ''.join(roman_num)

    @staticmethod
    def prompt_func(question_case) -> str:
        a = question_case['a']
        b = question_case['b']
        c = question_case['c']
        problem = (
            f"Convert the base-{a} number {c} to "
            f"{'Roman numerals' if b == 'R' else f'base {b}'}.\n\n"
            "Rules:\n"
            "- Input number may contain leading zeros\n"
            "- Roman numerals use subtractive notation (e.g., IV=4, IX=9)\n"
            "- Omit leading zeros in output except for zero value\n\n"
            "Put your final answer within [answer]...[/answer] tags."
        )
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
