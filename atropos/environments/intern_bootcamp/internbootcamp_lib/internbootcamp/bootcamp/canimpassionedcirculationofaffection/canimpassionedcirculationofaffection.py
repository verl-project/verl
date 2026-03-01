"""# 

### 谜题描述
Nadeko's birthday is approaching! As she decorated the room for the party, a long garland of Dianthus-shaped paper pieces was placed on a prominent part of the wall. Brother Koyomi will like it!

Still unsatisfied with the garland, Nadeko decided to polish it again. The garland has n pieces numbered from 1 to n from left to right, and the i-th piece has a colour si, denoted by a lowercase English letter. Nadeko will repaint at most m of the pieces to give each of them an arbitrary new colour (still denoted by a lowercase English letter). After this work, she finds out all subsegments of the garland containing pieces of only colour c — Brother Koyomi's favourite one, and takes the length of the longest among them to be the Koyomity of the garland.

For instance, let's say the garland is represented by \"kooomo\", and Brother Koyomi's favourite colour is \"o\". Among all subsegments containing pieces of \"o\" only, \"ooo\" is the longest, with a length of 3. Thus the Koyomity of this garland equals 3.

But problem arises as Nadeko is unsure about Brother Koyomi's favourite colour, and has swaying ideas on the amount of work to do. She has q plans on this, each of which can be expressed as a pair of an integer mi and a lowercase letter ci, meanings of which are explained above. You are to find out the maximum Koyomity achievable after repainting the garland according to each plan.

Input

The first line of input contains a positive integer n (1 ≤ n ≤ 1 500) — the length of the garland.

The second line contains n lowercase English letters s1s2... sn as a string — the initial colours of paper pieces on the garland.

The third line contains a positive integer q (1 ≤ q ≤ 200 000) — the number of plans Nadeko has.

The next q lines describe one plan each: the i-th among them contains an integer mi (1 ≤ mi ≤ n) — the maximum amount of pieces to repaint, followed by a space, then by a lowercase English letter ci — Koyomi's possible favourite colour.

Output

Output q lines: for each work plan, output one line containing an integer — the largest Koyomity achievable after repainting the garland according to it.

Examples

Input

6
koyomi
3
1 o
4 o
4 m


Output

3
6
5


Input

15
yamatonadeshiko
10
1 a
2 a
3 a
4 a
5 a
1 b
2 b
3 b
4 b
5 b


Output

3
4
5
7
8
1
2
3
4
5


Input

10
aaaaaaaaaa
2
10 b
10 z


Output

10
10

Note

In the first sample, there are three plans: 

  * In the first plan, at most 1 piece can be repainted. Repainting the \"y\" piece to become \"o\" results in \"kooomi\", whose Koyomity of 3 is the best achievable; 
  * In the second plan, at most 4 pieces can be repainted, and \"oooooo\" results in a Koyomity of 6; 
  * In the third plan, at most 4 pieces can be repainted, and \"mmmmmi\" and \"kmmmmm\" both result in a Koyomity of 5. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
iLength = int(raw_input())
sStr = raw_input()
#sStr = \"ab\"*750
questionList = []
iNum = int(raw_input())
for _ in xrange(iNum):
	iChange, sChar = raw_input().split()
	iChange = int(iChange)
	questionList.append((iChange, sChar))


hasCharDict = {}
for sChar in sStr:
	hasCharDict[sChar] = 1


ansDict = {} 
for sChar in \"abcdefghijklmnopqrstuvwxyz\":
	tempList = []
	for iIndex in xrange(iLength + 1):
		tempList.append(iIndex)
	ansDict[sChar] = tempList


def Solve(sString):
	posDict = {}
	for sChar in \"abcdefghijklmnopqrstuvwxyz\":
		if sChar in hasCharDict:
			posDict[sChar] = [0]
	for iIndex, sChar in enumerate(sString):
		if iIndex > 0:
			posDict[sChar].append(iIndex)
	for sChar in \"abcdefghijklmnopqrstuvwxyz\":
		if not sChar in hasCharDict:
			continue
		for iStart in posDict[sChar]:
			iEnd = iStart
			for iChange in xrange(iLength - iStart + 1):
				while iEnd < iLength and sString[iEnd] == sChar:
					iEnd += 1
				iMax = iEnd - iStart
				ansDict[sChar][iChange] = max(ansDict[sChar][iChange], iMax)
				if iEnd < iLength:
					iEnd += 1

Solve(sStr)
Solve(sStr[::-1])
for iChange, sChar in questionList:
	print ansDict[sChar][iChange]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import string
import random
from bootcamp import Basebootcamp

class Canimpassionedcirculationofaffectionbootcamp(Basebootcamp):
    def __init__(self, n=10, max_q=3, random_seed=None):
        super().__init__()
        self.n = n
        self.max_q = max_q
        self.rng = random.Random(random_seed)
    
    def case_generator(self):
        s = ''.join(self.rng.choice(string.ascii_lowercase) for _ in range(self.n))
        existing_chars = set(s)
        
        q = self.rng.randint(1, self.max_q)
        plans = []
        for _ in range(q):
            # 修正mi取值范围为1到n（原题中mi≥1）
            mi = self.rng.randint(1, self.n)
            
            # 增强生成逻辑：30%概率生成高频字符
            freq_chars = [c for c in existing_chars if s.count(c) > self.n//3] if existing_chars else []
            if freq_chars and self.rng.random() < 0.3:
                ci = self.rng.choice(freq_chars)
            elif existing_chars and self.rng.random() < 0.6:
                ci = self.rng.choice(list(existing_chars))
            else:
                ci = self.rng.choice(string.ascii_lowercase)
            
            plans.append((mi, ci))
        
        return {
            's': s,
            'plans': plans
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        plans = question_case['plans']
        prompt = [
            f"Garland: {s} (length {len(s)})",
            "Queries (max_repaints, target_color):"
        ]
        
        for i, (m, c) in enumerate(plans, 1):
            prompt.append(f"Query {i}: Allow {m} repaints to '{c}'")
        
        prompt.append(
            "Calculate maximum consecutive length for each query.\n"
            "Put all answers in one [answer] tag separated by spaces.\n"
            "Example: [answer]3 5 4[/answer]"
        )
        
        return '\n'.join(prompt)
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        try:
            # 支持多种分隔符（空格、逗号等）
            numbers = []
            for item in re.split(r'[\s,]+', matches[-1].strip()):
                if item.isdigit():
                    numbers.append(int(item))
            return numbers
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 增强类型检查
        if not isinstance(solution, list) or not all(isinstance(x, int) for x in solution):
            return False
        
        s = identity['s']
        plans = identity['plans']
        
        if len(solution) != len(plans):
            return False
        
        for ans, (mi, ci) in zip(solution, plans):
            # 添加边界检查
            if ans < 0 or ans > len(s):
                return False
            
            # 计算正确值
            correct = cls.compute_max_koyomity(s, ci, mi)
            if ans != correct:
                return False
        return True
    
    @staticmethod
    def compute_max_koyomity(s, c, m):
        def sliding_window(s_seq):
            max_len = left = changes = 0
            for right in range(len(s_seq)):
                if s_seq[right] != c:
                    changes += 1
                while changes > m:
                    if s_seq[left] != c:
                        changes -= 1
                    left += 1
                max_len = max(max_len, right - left + 1)
            return max_len
        
        # 添加预处理优化
        if m >= len(s) - s.count(c):
            return len(s)
        
        return max(sliding_window(s), sliding_window(s[::-1]))
