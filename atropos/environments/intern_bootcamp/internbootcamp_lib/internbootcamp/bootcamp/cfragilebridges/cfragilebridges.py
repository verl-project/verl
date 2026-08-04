"""# 

### 谜题描述
You are playing a video game and you have just reached the bonus level, where the only possible goal is to score as many points as possible. Being a perfectionist, you've decided that you won't leave this level until you've gained the maximum possible number of points there.

The bonus level consists of n small platforms placed in a line and numbered from 1 to n from left to right and (n - 1) bridges connecting adjacent platforms. The bridges between the platforms are very fragile, and for each bridge the number of times one can pass this bridge from one of its ends to the other before it collapses forever is known in advance.

The player's actions are as follows. First, he selects one of the platforms to be the starting position for his hero. After that the player can freely move the hero across the platforms moving by the undestroyed bridges. As soon as the hero finds himself on a platform with no undestroyed bridge attached to it, the level is automatically ended. The number of points scored by the player at the end of the level is calculated as the number of transitions made by the hero between the platforms. Note that if the hero started moving by a certain bridge, he has to continue moving in the same direction until he is on a platform.

Find how many points you need to score to be sure that nobody will beat your record, and move to the next level with a quiet heart.

Input

The first line contains a single integer n (2 ≤ n ≤ 105) — the number of platforms on the bonus level. The second line contains (n - 1) integers ai (1 ≤ ai ≤ 109, 1 ≤ i < n) — the number of transitions from one end to the other that the bridge between platforms i and i + 1 can bear.

Output

Print a single integer — the maximum number of points a player can get on the bonus level.

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5
2 1 2 1


Output

5

Note

One possibility of getting 5 points in the sample is starting from platform 3 and consequently moving to platforms 4, 3, 2, 1 and 2. After that the only undestroyed bridge is the bridge between platforms 4 and 5, but this bridge is too far from platform 2 where the hero is located now.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
dpL = [-1 for i in range(100005)]
dpR = [-1 for i in range(100005)]
dpbackL = [-1 for i in range(100005)]
dpbackR = [-1 for i in range(100005)]
global ip
N = input()
ip = map(int,raw_input().split())
x = N-1
for x in range(N-1,-1,-1) :
	if (x == N-1) :
		dpbackR[x] = 0
		continue
	if (ip[x] <= 1):
		dpbackR[x] = 0
		continue
	if (ip[x] % 2 == 0):
		dpbackR[x] = ip[x]
	else :
		dpbackR[x] = ip[x]-1
	dpbackR[x] = dpbackR[x] + dpbackR[x+1]
for x in range(N-1,-1,-1):
	if (x == N-1) :
		dpR[x] = 0
		continue
	if (ip[x] % 2 == 0):
		dpR[x] = ip[x] - 1
	else:
		dpR[x] = ip[x]
	dpR[x] = max(dpR[x] + dpR[x+1],dpbackR[x])
	
for x in range(N) :
	if (x == 0) :
		dpbackL[x] = 0
		continue
	if (ip[x-1] <= 1):
		dpbackL[x] = 0
		continue
	if (ip[x-1] % 2 == 0):
		dpbackL[x] = ip[x-1]
	else :
		dpbackL[x] = ip[x-1]-1
	dpbackL[x] = dpbackL[x] + dpbackL[x-1]
for x in range(N) :
	if (x == 0):
		dpL[x] = 0
		continue
	if (ip[x-1] % 2 == 0):
		dpL[x] = ip[x-1] - 1
	else :
		dpL[x] = ip[x-1]
	dpL[x] = max(dpL[x] + dpL[x-1], dpbackL[x])

best = 0
for i in range (N):
	best = max(best, dpbackL[i] + max(dpbackR[i],dpR[i]))
	best = max(best, dpbackR[i] + max(dpbackL[i],dpL[i]))
print best

'''
def FBR(x):
	if (x == N-1) :
		return 0
	if (ip[x] <= 1) :
		return 0
	if (dpbackR[x] != -1):
		return dpbackR[x]
	if (ip[x] % 2 == 0):
		dpbackR[x] = ip[x]
	else :
		dpbackR[x] = ip[x]-1
	dpbackR[x] = dpbackR[x] + FBR(x+1)
	return dpbackR[x]

def FBL(x):
	if (x == 0) :
		return 0
	if (ip[x-1] <= 1) : 
		return 0
	if (dpbackL[x] != -1):
		return dpbackL[x]
	if (ip[x-1] % 2 == 0):
		dpbackL[x] = ip[x-1]
	else :
		dpbackL[x] = ip[x-1]-1
	dpbackL[x] = dpbackL[x] + FBL(x-1)
	return dpbackL[x]

def FR(x):
	if (x == N-1) :
		return 0
	if (dpR[x] != -1):
		return dpR[x]
	if (ip[x] % 2 == 0):
		dpR[x] = ip[x] - 1
	else:
		dpR[x] = ip[x]
	dpR[x] = max(dpR[x] + FR(x+1),FBR(x))
	return dpR[x]

def FL(x):
	if (x == 0) :
		return 0
	if (dpL[x] != -1) :
		return dpL[x]
	if (ip[x-1] % 2 == 0) :
		dpL[x] = ip[x-1] - 1
	else :
		dpL[x] = ip[x-1]
	dpL[x] = max(dpL[x] + FL(x-1), FBL(x))
	return dpL[x]

'''
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cfragilebridgesbootcamp(Basebootcamp):
    def __init__(self, max_n=10, min_n=2, ai_max=10**9):
        self.max_n = max_n
        self.min_n = min_n
        self.ai_max = ai_max

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        ai = [random.randint(1, self.ai_max) for _ in range(n-1)]
        return {
            'n': n,
            'ai': ai,
            'correct_answer': self.compute_max_points(n, ai)
        }

    @staticmethod
    def compute_max_points(n, a):
        if n < 2:
            return 0
        dpbackR = [0] * n
        for x in range(n-2, -1, -1):
            if a[x] <= 1:
                dpbackR[x] = 0
            else:
                val = a[x] if a[x] % 2 == 0 else a[x] - 1
                dpbackR[x] = val + dpbackR[x+1]
        
        dpR = [0] * n
        for x in range(n-2, -1, -1):
            if a[x] % 2 == 0:
                option1 = (a[x] - 1) + dpR[x+1]
            else:
                option1 = a[x] + dpR[x+1]
            option2 = dpbackR[x]
            dpR[x] = max(option1, option2)
        
        dpbackL = [0] * n
        for x in range(1, n):
            if a[x-1] <= 1:
                dpbackL[x] = 0
            else:
                val = a[x-1] if a[x-1] % 2 == 0 else a[x-1] - 1
                dpbackL[x] = val + dpbackL[x-1]
        
        dpL = [0] * n
        for x in range(1, n):
            if a[x-1] % 2 == 0:
                option1 = (a[x-1] - 1) + dpL[x-1]
            else:
                option1 = a[x-1] + dpL[x-1]
            option2 = dpbackL[x]
            dpL[x] = max(option1, option2)
        
        best = 0
        for i in range(n):
            best = max(best, dpbackL[i] + max(dpbackR[i], dpR[i]))
            best = max(best, dpbackR[i] + max(dpbackL[i], dpL[i]))
        return best

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        ai = question_case['ai']
        input_str = f"{n}\n{' '.join(map(str, ai))}"
        prompt = f"""You are playing a video game bonus level with platforms connected by bridges. Each bridge has a durability indicating how many times it can be crossed.

Task:
Find the maximum transitions possible before bridges collapse.

Input Format:
{n}
{' '.join(map(str, ai))}

Answer format: [answer]integer[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
