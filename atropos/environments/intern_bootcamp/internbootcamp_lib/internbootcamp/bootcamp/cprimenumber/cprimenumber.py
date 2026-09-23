"""# 

### 谜题描述
Simon has a prime number x and an array of non-negative integers a1, a2, ..., an.

Simon loves fractions very much. Today he wrote out number <image> on a piece of paper. After Simon led all fractions to a common denominator and summed them up, he got a fraction: <image>, where number t equals xa1 + a2 + ... + an. Now Simon wants to reduce the resulting fraction. 

Help him, find the greatest common divisor of numbers s and t. As GCD can be rather large, print it as a remainder after dividing it by number 1000000007 (109 + 7).

Input

The first line contains two positive integers n and x (1 ≤ n ≤ 105, 2 ≤ x ≤ 109) — the size of the array and the prime number.

The second line contains n space-separated integers a1, a2, ..., an (0 ≤ a1 ≤ a2 ≤ ... ≤ an ≤ 109). 

Output

Print a single number — the answer to the problem modulo 1000000007 (109 + 7).

Examples

Input

2 2
2 2


Output

8


Input

3 3
1 2 3


Output

27


Input

2 2
29 29


Output

73741817


Input

4 5
0 0 0 0


Output

1

Note

In the first sample <image>. Thus, the answer to the problem is 8.

In the second sample, <image>. The answer to the problem is 27, as 351 = 13·27, 729 = 27·27.

In the third sample the answer to the problem is 1073741824 mod 1000000007 = 73741817.

In the fourth sample <image>. Thus, the answer to the problem is 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# coding: utf-8
# Laybson Plismenn / UFCG - 2015

mod = 1000000007

def min(a,b):
	if a<b :
		return a
	else:
		return b
		
def Pow(a,b):
	d=1
	t=a
	while b>0:
		if b%2==1:
			d=d*t%mod
		b/=2
		t=t*t%mod
	return d

a=[]
b=[]
vis=[]
sum=0
n,x=map(int,raw_input().split())
a=(map(int,raw_input().split()))
for i in range(len(a)):
	sum += a[i]
for i in range(len(a)):
	b.append(sum-a[i])
	vis.append(0)

Max=sum+10
ans=1
while True:
	Ma = Max
	for i in range(len(b)):
		if vis[i]==0 :
			Ma=min(Ma,b[i])
	if (Ma==0 or Ma==Max):
		break
	ans *= Pow(x,Ma)
	ans %= mod
	num=0
	for i in range(len(b)):
		b[i] -= Ma
		if(b[i]==0):
			num+=1
	sum = sum-Ma
	if (sum<=0 or num%x != 0):
		break
	else :
		p=0
		for i in range(len(b)):
			if(b[i]==0):
				if(p < num/x):
					b[i]=1
					p+=1
				else :
					vis[i]=1

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

mod = 10**9 + 7

def calculate_gcd_mod(n, x, a_list):
    sum_total = sum(a_list)
    b = [sum_total - ai for ai in a_list]
    vis = [False] * n
    ans = 1
    while True:
        current_min = None
        for i in range(n):
            if not vis[i] and (current_min is None or b[i] < current_min):
                current_min = b[i]
        if current_min is None or current_min == 0:
            break
        ans = ans * pow(x, current_min, mod) % mod
        count = 0
        new_sum = sum_total - current_min
        for i in range(n):
            if not vis[i]:
                b[i] -= current_min
                if b[i] == 0:
                    count += 1
        sum_total = new_sum
        if sum_total <= 0 or count % x != 0:
            break
        else:
            target = count // x
            p = 0
            for i in range(n):
                if not vis[i] and b[i] == 0:
                    if p < target:
                        b[i] = 1
                        p += 1
                    else:
                        vis[i] = True
    return ans % mod

class Cprimenumberbootcamp(Basebootcamp):
    def __init__(self, max_n=5, x_primes=None, a_max=10):
        super().__init__()
        self.max_n = max_n
        self.x_primes = x_primes if x_primes is not None else [2, 3, 5, 7, 11]
        self.a_max = a_max

    def case_generator(self):
        n = random.randint(1, self.max_n)
        x = random.choice(self.x_primes)
        a = sorted(random.randint(0, self.a_max) for _ in range(n))
        correct_answer = calculate_gcd_mod(n, x, a)
        return {
            'n': n,
            'x': x,
            'a': a,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        x = question_case['x']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        return f"""Simon has a prime number x and an array of non-negative integers. Your task is to compute the GCD of the fraction's numerator and denominator after summing 1/x^a_i for all elements. 

Input:
First line: {n} {x} (n and the prime x)
Second line: {a_str} (non-decreasing array)

Calculate the GCD modulo 1,000,000,007. Put your final answer within [answer] and [/answer] tags. Example: [answer]42[/answer]."""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            number_str = ''.join(c for c in last_answer if c.isdigit())
            return int(number_str) if number_str else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        return solution == identity['correct_answer']
