"""# 

### 谜题描述
Andrew and Jerry are playing a game with Harry as the scorekeeper. The game consists of three rounds. In each round, Andrew and Jerry draw randomly without replacement from a jar containing n balls, each labeled with a distinct positive integer. Without looking, they hand their balls to Harry, who awards the point to the player with the larger number and returns the balls to the jar. The winner of the game is the one who wins at least two of the three rounds.

Andrew wins rounds 1 and 2 while Jerry wins round 3, so Andrew wins the game. However, Jerry is unhappy with this system, claiming that he will often lose the match despite having the higher overall total. What is the probability that the sum of the three balls Jerry drew is strictly higher than the sum of the three balls Andrew drew?

Input

The first line of input contains a single integer n (2 ≤ n ≤ 2000) — the number of balls in the jar.

The second line contains n integers ai (1 ≤ ai ≤ 5000) — the number written on the ith ball. It is guaranteed that no two balls have the same number.

Output

Print a single real value — the probability that Jerry has a higher total, given that Andrew wins the first two rounds and Jerry wins the third. Your answer will be considered correct if its absolute or relative error does not exceed 10 - 6. 

Namely: let's assume that your answer is a, and the answer of the jury is b. The checker program will consider your answer correct, if <image>.

Examples

Input

2
1 2


Output

0.0000000000


Input

3
1 2 10


Output

0.0740740741

Note

In the first case, there are only two balls. In the first two rounds, Andrew must have drawn the 2 and Jerry must have drawn the 1, and vice versa in the final round. Thus, Andrew's sum is 5 and Jerry's sum is 4, so Jerry never has a higher total.

In the second case, each game could've had three outcomes — 10 - 2, 10 - 1, or 2 - 1. Jerry has a higher total if and only if Andrew won 2 - 1 in both of the first two rounds, and Jerry drew the 10 in the last round. This has probability <image>.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n , m = input() , map(int,raw_input().split())
a = {}
for i in xrange(n-1):
	for j in xrange(i+1,n):
		x = abs(m[i]-m[j])
		if x in a:
			a[x]+=1
		else:
			a[x]=1

d = [i for i in a]
b = [0]*10005
for i in xrange(len(d)):
	for j in xrange(i,len(d)):
		if d[i]==d[j]:
			b[d[i]+d[j]] += a[d[i]]*a[d[j]]
 		else:
			b[d[i]+d[j]] += a[d[i]]*a[d[j]]*2

for i in xrange(1,len(b)): b[i]=b[i]+b[i-1]
ans=0
for i in xrange(n-1):
	for j in xrange(i+1,n):
		ans+=b[abs(m[i]-m[j])-1]

den = (n*(n-1)/2)**3
print ans/float(den)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def calculate_probability(n, a_list):
    a = {}
    for i in range(n-1):
        for j in range(i+1, n):
            x = abs(a_list[i] - a_list[j])
            a[x] = a.get(x, 0) + 1

    d = list(a.keys())
    b = [0] * 10005

    for i in range(len(d)):
        for j in range(i, len(d)):
            key_i = d[i]
            key_j = d[j]
            sum_key = key_i + key_j
            contribution = a[key_i] * a[key_j]
            if key_i != key_j:
                contribution *= 2
            if sum_key < len(b):
                b[sum_key] += contribution

    for i in range(1, len(b)):
        b[i] += b[i-1]

    ans = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s = abs(a_list[i] - a_list[j])
            if s - 1 >= 0 and s - 1 < len(b):
                ans += b[s - 1]

    den = (n * (n-1) // 2) ** 3
    return ans / den if den != 0 else 0.0

def is_close(a, b, rel_tol=1e-6, abs_tol=1e-6):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class Djerrysprotestbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10, a_min=1, a_max=5000):
        self.n_min = n_min
        self.n_max = n_max
        self.a_min = a_min
        self.a_max = a_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        possible = list(range(self.a_min, self.a_max + 1))
        a = random.sample(possible, n)
        return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        problem = (
            "Andrew 和 Jerry 正在玩一个游戏，Harry 担任裁判。游戏进行三轮，每轮两人从罐子中随机抽取不同的球（球上有唯一正整数）。\n"
            "规则：\n"
            "1. 每轮两人抽球后，数字大者获胜，球放回罐子\n"
            "2. 先赢两轮者赢得比赛\n\n"
            f"当前罐中共有 {n} 个球，数字分别为：{', '.join(map(str, sorted(a)))}。\n"
            "已知 Andrew 赢了前两轮，Jerry 赢了第三轮。求 Jerry 三轮数字之和严格大于 Andrew 的概率。\n"
            "要求：答案保留至小数点后10位，误差不超过1e-6。将最终答案放在 [answer] 和 [/answer] 标签之间。\n"
            "示例：若答案为0.123456，则写为：[answer]0.1234560000[/answer]"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_value = matches[-1].strip()
        try:
            return float(re.sub(r'[^\d.eE-]', '', last_value))
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            user_ans = float(solution)
        except:
            return False
        
        n = identity['n']
        a = identity['a']
        correct = calculate_probability(n, a)
        return is_close(user_ans, correct)
