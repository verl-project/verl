"""# 

### 谜题描述
After all the events in Orlando we all know, Sasha and Roma decided to find out who is still the team's biggest loser. Thankfully, Masha found somewhere a revolver with a rotating cylinder of n bullet slots able to contain exactly k bullets, now the boys have a chance to resolve the problem once and for all. 

Sasha selects any k out of n slots he wishes and puts bullets there. Roma spins the cylinder so that every of n possible cylinder's shifts is equiprobable. Then the game starts, the players take turns, Sasha starts: he puts the gun to his head and shoots. If there was no bullet in front of the trigger, the cylinder shifts by one position and the weapon is given to Roma for make the same move. The game continues until someone is shot, the survivor is the winner. 

Sasha does not want to lose, so he must choose slots for bullets in such a way as to minimize the probability of its own loss. Of all the possible variant he wants to select the lexicographically minimal one, where an empty slot is lexicographically less than a charged one. 

More formally, the cylinder of n bullet slots able to contain k bullets can be represented as a string of n characters. Exactly k of them are \"X\" (charged slots) and the others are \".\" (uncharged slots). 

Let us describe the process of a shot. Suppose that the trigger is in front of the first character of the string (the first slot). If a shot doesn't kill anyone and the cylinder shifts, then the string shifts left. So the first character becomes the last one, the second character becomes the first one, and so on. But the trigger doesn't move. It will be in front of the first character of the resulting string.

Among all the strings that give the minimal probability of loss, Sasha choose the lexicographically minimal one. According to this very string, he charges the gun. You have to help Sasha to charge the gun. For that, each xi query must be answered: is there a bullet in the positions xi?

Input

The first line contains three integers n, k and p (1 ≤ n ≤ 1018, 0 ≤ k ≤ n, 1 ≤ p ≤ 1000) — the number of slots in the cylinder, the number of bullets and the number of queries. Then follow p lines; they are the queries. Each line contains one integer xi (1 ≤ xi ≤ n) the number of slot to describe.

Please do not use the %lld specificator to read or write 64-bit numbers in С++. It is preferred to use cin, cout streams or the %I64d specificator.

Output

For each query print \".\" if the slot should be empty and \"X\" if the slot should be charged.

Examples

Input

3 1 3
1
2
3


Output

..X

Input

6 3 6
1
2
3
4
5
6


Output

.X.X.X

Input

5 2 5
1
2
3
4
5


Output

...XX

Note

The lexicographical comparison of is performed by the < operator in modern programming languages. The a string is lexicographically less that the b string, if there exists such i (1 ≤ i ≤ n), that ai < bi, and for any j (1 ≤ j < i) aj = bj.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m,k=map(int,raw_input().split())

first=n-(2*m-1)

if first%2==0:
    retr=1
    first+=1
else:
    retr=0

ans=\"\"

for i in xrange(k):
    q=int(raw_input())
    if first>0:
        if retr==1 and q==n and m>0: ans+=\"X\"
        elif q<=first: ans+=\".\"
        elif q%2==0: ans+=\"X\"
        else: ans+=\".\"
    else:
        if q%2==1 and q<(n-m)*2:ans+=\".\"
        else: ans+=\"X\"
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random

class Crussianroulettebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = max(1, params.get('n_min', 1))
        self.n_max = max(self.n_min, params.get('n_max', 20))
        self.k_min = max(0, params.get('k_min', 0))
        self.k_max = params.get('k_max')
        self.p_min = max(1, params.get('p_min', 1))
        self.p_max = max(self.p_min, params.get('p_max', 5))
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        k_upper = min(n, self.k_max) if self.k_max is not None else n
        k = random.randint(max(self.k_min, 0), k_upper)
        p = random.randint(self.p_min, self.p_max)
        # 生成可能包含重复的查询位置
        queries = [random.randint(1, n) for _ in range(p)]  
        return {'n': n, 'k': k, 'queries': queries}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""作为转轮手枪专家，请解决以下配置问题：

# 基础参数
- 总槽位：{question_case['n']}
- 子弹数：{question_case['k']}
- 需查询的槽位位置：{question_case['queries']}

# 配置要求
1. 找到最小化Sasha死亡概率的配置方案
2. 当存在多个最优方案时选择字典序最小的（'.' < 'X'）

# 转轮运作规则
- 每次射击后若无子弹，转轮左移1位
- 射击顺序：Sasha -> Roma -> Sasha... 交替进行

# 输出格式
将最终答案用[answer][/answer]包裹，例如查询位置2和5时为：[answer].X[/answer]

请严格按照要求输出："""
    
    @staticmethod
    def extract_output(output):
        # 增强模式匹配，允许前后空格
        matches = re.findall(r'\[answer\]\s*([X.]+?)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1].upper().replace(' ', '') if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 精确实现参考代码逻辑
        n = identity['n']
        k = identity['k']
        queries = identity['queries']
        
        if k == 0:
            correct = '.' * len(queries)
            return solution == correct
        
        first = n - (2*k -1)
        retr = 0
        if first >= 0:
            if first % 2 == 0:
                retr = 1
                first += 1
            else:
                retr = 0
        else:
            first = 0
        
        res = []
        for q in queries:
            if first > 0:
                if retr and q == n and k > 0:
                    res.append('X')
                elif q <= first:
                    res.append('.')
                else:
                    res.append( 'X' if (q - first) % 2 == 1 else '.' )
            else:
                threshold = 2*(n - k)
                res.append( '.' if q % 2 == 1 and q <= threshold else 'X' )
        
        return solution == ''.join(res)
