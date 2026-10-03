"""# 

### 谜题描述
Since Sonya is interested in robotics too, she decided to construct robots that will read and recognize numbers.

Sonya has drawn n numbers in a row, a_i is located in the i-th position. She also has put a robot at each end of the row (to the left of the first number and to the right of the last number). Sonya will give a number to each robot (they can be either same or different) and run them. When a robot is running, it is moving toward to another robot, reading numbers in the row. When a robot is reading a number that is equal to the number that was given to that robot, it will turn off and stay in the same position.

Sonya does not want robots to break, so she will give such numbers that robots will stop before they meet. That is, the girl wants them to stop at different positions so that the first robot is to the left of the second one.

For example, if the numbers [1, 5, 4, 1, 3] are written, and Sonya gives the number 1 to the first robot and the number 4 to the second one, the first robot will stop in the 1-st position while the second one in the 3-rd position. In that case, robots will not meet each other. As a result, robots will not be broken. But if Sonya gives the number 4 to the first robot and the number 5 to the second one, they will meet since the first robot will stop in the 3-rd position while the second one is in the 2-nd position.

Sonya understands that it does not make sense to give a number that is not written in the row because a robot will not find this number and will meet the other robot.

Sonya is now interested in finding the number of different pairs that she can give to robots so that they will not meet. In other words, she wants to know the number of pairs (p, q), where she will give p to the first robot and q to the second one. Pairs (p_i, q_i) and (p_j, q_j) are different if p_i≠ p_j or q_i≠ q_j.

Unfortunately, Sonya is busy fixing robots that broke after a failed launch. That is why she is asking you to find the number of pairs that she can give to robots so that they will not meet.

Input

The first line contains a single integer n (1≤ n≤ 10^5) — the number of numbers in a row.

The second line contains n integers a_1, a_2, …, a_n (1≤ a_i≤ 10^5) — the numbers in a row.

Output

Print one number — the number of possible pairs that Sonya can give to robots so that they will not meet.

Examples

Input

5
1 5 4 1 3


Output

9


Input

7
1 2 1 1 1 3 2


Output

7

Note

In the first example, Sonya can give pairs (1, 1), (1, 3), (1, 4), (1, 5), (4, 1), (4, 3), (5, 1), (5, 3), and (5, 4).

In the second example, Sonya can give pairs (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), and (3, 2).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from fractions import gcd
from math import factorial, ceil, sqrt, atan2, log, pi, e, asin,acos, cos, sin, floor
from itertools import *
from fractions import Fraction
import string
import copy
import random
import bisect
from decimal import *
from collections import deque

def id_generator(size=20, chars=string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
 
def mp():
    return map(int,str(raw_input()).split())

'''
n,m=mp()
l=[]
for i in range(m):
	a,b=mp()
	l+=[[a,b]]
l.sort()
s='0'*n
'''

n=input()
l=list(mp())
d={}
a=[]
distinct=len(set(l))
#print distinct
for i in range(n):
	try:
		if d[l[i]]:
			d[l[i]][0]+=1
	except:
		d[l[i]]=[1,0]
ans=0
for i in range(n):
	d[l[i]][0]-=1
	#print distinct
	if d[l[i]][0]==0:
		distinct-=1
	#print distinct,d,d[l[i]]
	if d[l[i]][1]==0:
		ans+=distinct
		d[l[i]][1]=1
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import bisect
from collections import defaultdict
from bootcamp import Basebootcamp

class Csonyaandrobotsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=1000, max_value=100000):
        self.min_n = min_n
        self.max_n = max_n
        self.max_value = max_value
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        a = [random.randint(1, self.max_value) for _ in range(n)]
        return {"n": n, "a": a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a_str = ' '.join(map(str, question_case['a']))
        prompt = f"""索尼娅在一行中放置了{n}个数字：{a_str}。她在行的两端各放置了一个机器人。左边的机器人被赋予数字p，右边的被赋予数字q。两个机器人开始相向移动：

- 左边机器人向右移动，在找到第一个p时停在对应位置
- 右边机器人向左移动，在找到第一个q时停在对应位置

有效条件：左边机器人位置必须严格小于右边机器人位置，且p和q必须存在于数组中。

请计算所有有效(p, q)对的数量。答案放入[answer][/answer]。例如，对于输入：
5
1 5 4 1 3

正确格式：[answer]9[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        first_occurrence = {}
        for idx, num in enumerate(a):
            if num not in first_occurrence:
                first_occurrence[num] = idx
        
        last_occurrence = {}
        for idx in reversed(range(len(a))):
            num = a[idx]
            if num not in last_occurrence:
                last_occurrence[num] = idx
        
        last_positions = sorted(last_occurrence.values())
        total = 0
        for p in first_occurrence:
            p_first = first_occurrence[p]
            cnt = len(last_positions) - bisect.bisect_right(last_positions, p_first)
            total += cnt
        
        return solution == total
