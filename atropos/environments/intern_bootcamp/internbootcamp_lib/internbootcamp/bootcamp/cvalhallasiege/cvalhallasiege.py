"""# 

### 谜题描述
Ivar the Boneless is a great leader. He is trying to capture Kattegat from Lagertha. The war has begun and wave after wave Ivar's warriors are falling in battle.

Ivar has n warriors, he places them on a straight line in front of the main gate, in a way that the i-th warrior stands right after (i-1)-th warrior. The first warrior leads the attack.

Each attacker can take up to a_i arrows before he falls to the ground, where a_i is the i-th warrior's strength.

Lagertha orders her warriors to shoot k_i arrows during the i-th minute, the arrows one by one hit the first still standing warrior. After all Ivar's warriors fall and all the currently flying arrows fly by, Thor smashes his hammer and all Ivar's warriors get their previous strengths back and stand up to fight again. In other words, if all warriors die in minute t, they will all be standing to fight at the end of minute t.

The battle will last for q minutes, after each minute you should tell Ivar what is the number of his standing warriors.

Input

The first line contains two integers n and q (1 ≤ n, q ≤ 200 000) — the number of warriors and the number of minutes in the battle.

The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 10^9) that represent the warriors' strengths.

The third line contains q integers k_1, k_2, …, k_q (1 ≤ k_i ≤ 10^{14}), the i-th of them represents Lagertha's order at the i-th minute: k_i arrows will attack the warriors.

Output

Output q lines, the i-th of them is the number of standing warriors after the i-th minute.

Examples

Input

5 5
1 2 1 2 1
3 10 1 1 1


Output

3
5
4
4
3


Input

4 4
1 2 3 4
9 1 10 6


Output

1
4
4
1

Note

In the first example: 

  * after the 1-st minute, the 1-st and 2-nd warriors die. 
  * after the 2-nd minute all warriors die (and all arrows left over are wasted), then they will be revived thus answer is 5 — all warriors are alive. 
  * after the 3-rd minute, the 1-st warrior dies. 
  * after the 4-th minute, the 2-nd warrior takes a hit and his strength decreases by 1. 
  * after the 5-th minute, the 2-nd warrior dies. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
from atexit import register
from io import BytesIO
sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))
input = lambda: sys.stdin.readline().rstrip('\r\n')
raw_input = lambda: sys.stdin.readline().rstrip('\r\n')


n,q =  map(int,raw_input().split(\" \"))
ai  =  map(int,raw_input().split(\" \"))
ki  =  map(int,raw_input().split(\" \"))
def bs(arr,v):
	low = 0
	high = n-1
	while low <= high:
		mid = (low +high)/2
		if arr[mid]>v:
			high = mid-1
		else:
			low = mid+1
	return low-1

maxv = sum(ai)
psum = [0]*(1+n)
for i in range(1,n+1):
	psum[i] = psum[i-1]+ai[i-1]

# print psum

s = 0
for i in range(q):
	s += ki[i]
	if s >= maxv:
		s = 0
		print n
	else:
		p = bs(psum,s)
		# print i,s,p
		print n-p
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cvalhallasiegebootcamp(Basebootcamp):
    def __init__(self, n=5, q=5, max_a=10**9, max_k=10**14):
        self.n = n
        self.q = q
        self.max_a = max_a
        self.max_k = max_k
    
    def case_generator(self):
        a = [random.randint(1, self.max_a) for _ in range(self.n)]
        sum_a = sum(a)
        
        k = []
        for _ in range(self.q):
            if random.random() < 0.2:
                ki = random.randint(sum_a//2, self.max_k)
            else:
                upper = min(sum_a*2, self.max_k)
                ki = random.randint(1, upper)
            k.append(ki)
        
        return {
            'n': self.n,
            'q': self.q,
            'a': a,
            'k': k,
            'sum_a': sum_a
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""Ivar的战士排成一列面对城堡大门。每个战士能承受的箭数为a_i。战斗持续q分钟，每分钟射出k_i支箭，箭依次攻击存活的第一个战士。当所有战士倒下时，当前分钟剩余箭矢报废，战士立即复活。请计算每分钟后的存活战士数。

输入格式：
第一行：n={question_case['n']} q={question_case['q']}
第二行：{' '.join(map(str, question_case['a']))}
第三行：{' '.join(map(str, question_case['k']))}

输出{question_case['q']}行，每行一个整数表示存活数。请将答案严格按以下格式放置：

[answer]
示例答案（共{question_case['q']}行）：
3
5
4
...
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_answer = matches[-1].strip()
        solution = []
        for line in last_answer.split('\n'):
            line = line.strip()
            if line and line.isdigit():
                solution.append(int(line))
        
        return solution  # 移除了错误的条件判断

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        k_list = identity['k']
        sum_a = identity['sum_a']
        n = identity['n']
        q = identity['q']
        
        if len(solution) != q:
            return False
        
        psum = [0] * (n + 1)
        for i in range(1, n+1):
            psum[i] = psum[i-1] + a[i-1]
        
        def binary_search(s_val):
            low, high = 0, n
            while low <= high:
                mid = (low + high) // 2
                if psum[mid] > s_val:
                    high = mid - 1
                else:
                    low = mid + 1
            return high  # 正确的二分查找逻辑
        
        s = 0
        correct = []
        for k in k_list:
            s += k
            if s >= sum_a:
                correct.append(n)
                s = 0
            else:
                p = binary_search(s)
                correct.append(n - p)
        
        return solution == correct
