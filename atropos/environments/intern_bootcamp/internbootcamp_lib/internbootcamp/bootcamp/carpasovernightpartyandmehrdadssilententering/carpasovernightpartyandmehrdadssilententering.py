"""# 

### 谜题描述
Note that girls in Arpa’s land are really attractive.

Arpa loves overnight parties. In the middle of one of these parties Mehrdad suddenly appeared. He saw n pairs of friends sitting around a table. i-th pair consisted of a boy, sitting on the ai-th chair, and his girlfriend, sitting on the bi-th chair. The chairs were numbered 1 through 2n in clockwise direction. There was exactly one person sitting on each chair.

<image>

There were two types of food: Kooft and Zahre-mar. Now Mehrdad wonders, was there any way to serve food for the guests such that: 

  * Each person had exactly one type of food, 
  * No boy had the same type of food as his girlfriend, 
  * Among any three guests sitting on consecutive chairs, there was two of them who had different type of food. Note that chairs 2n and 1 are considered consecutive. 



Find the answer for the Mehrdad question. If it was possible, find some arrangement of food types that satisfies the conditions.

Input

The first line contains an integer n (1 ≤ n ≤ 105) — the number of pairs of guests.

The i-th of the next n lines contains a pair of integers ai and bi (1 ≤ ai, bi ≤ 2n) — the number of chair on which the boy in the i-th pair was sitting and the number of chair on which his girlfriend was sitting. It's guaranteed that there was exactly one person sitting on each chair. 

Output

If there is no solution, print -1.

Otherwise print n lines, the i-th of them should contain two integers which represent the type of food for the i-th pair. The first integer in the line is the type of food the boy had, and the second integer is the type of food the girl had. If someone had Kooft, print 1, otherwise print 2.

If there are multiple solutions, print any of them.

Example

Input

3
1 4
2 5
3 6


Output

1 2
2 1
1 2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=int(raw_input())
g=[[]for i in range(n+n)]
c=[-1 for i in range(n+n)]
e=[]
for i in range(n):
    u,v=map(int,raw_input().split())
    u-=1
    v-=1
    e.append([u,v])
    g[u].append(v)
    g[v].append(u)
for i in range(n):
    g[2*i].append(2*i+1)
    g[2*i+1].append(2*i)
q=[-1 for i in range(2*n)]
for i in range(2*n):
    if c[i]!=-1:continue
    h,t=0,1
    q[0]=i
    c[i]=0
    while h<t:
        u=q[h]
        h+=1
        for v in g[u]:
            if c[v]==-1:
                c[v]=c[u]^1
                q[t]=v
                t+=1
for x in e: print \"{0} {1}\".format(c[x[0]]+1,c[x[1]]+1)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re

class Carpasovernightpartyandmehrdadssilententeringbootcamp(Basebootcamp):
    def __init__(self, n=3):
        """
        初始化谜题参数，默认椅子对数为3
        :param n: 椅子对数 (1 ≤ n ≤ 1e5)
        """
        self.n = n

    def case_generator(self):
        """
        生成保证有解的合法案例
        返回: {'n': n, 'pairs': [(a1, b1), (a2, b2), ...]}
        """
        n = self.n
        # 生成奇数和偶数位置的椅子
        odd_chairs = list(range(1, 2*n+1, 2))  # 奇数座位：1,3,5,...
        even_chairs = list(range(2, 2*n+1, 2)) # 偶数座位：2,4,6,...

        # 打乱后配对确保奇偶匹配
        random.shuffle(odd_chairs)
        random.shuffle(even_chairs)
        
        pairs = []
        for o, e in zip(odd_chairs, even_chairs):
            # 随机交换男女位置
            if random.choice([True, False]):
                pairs.append((o, e))
            else:
                pairs.append((e, o))

        return {
            'n': n,
            'pairs': pairs
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        pairs = question_case['pairs']
        
        problem = "You are trying to solve Carpasovernightpartyandmehrdadssilententering's food assignment problem at a circular party table.\n\n"
        problem += "Rules:\n"
        problem += "1. Each guest must choose either Kooft (1) or Zahre-mar (2)\n"
        problem += "2. Boys and their girlfriends must have different food types\n"
        problem += "3. Any three consecutive guests (including wrap-around) must not all have the same type\n\n"
        problem += f"Problem Instance (n={n}):\n"
        problem += f"{n}\n" + "\n".join(f"{a} {b}" for a, b in pairs) + "\n\n"
        problem += "Output format: print -1 if impossible. Otherwise, print n lines with boy's and girl's food types.\n"
        problem += "Example:\n[answer]\n1 2\n2 1\n1 2\n[/answer]\n"
        problem += "Put your final answer between [answer] and [/answer] tags."
        return problem

    @staticmethod
    def extract_output(output):
        # 严格匹配最后出现的答案块
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        answer_block = matches[-1].strip()
        if answer_block == '-1':
            return -1
        
        solution = []
        for line in answer_block.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                return None
            try:
                a, b = int(parts[0]), int(parts[1])
                if {a, b} <= {1, 2} and a != b:
                    solution.append((a, b))
                else:
                    return None
            except:
                return None
        
        return solution if solution else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证答案是否符合所有约束
        if solution == -1:
            return False  # 我们的生成器保证有解
        
        n = identity['n']
        pairs = identity['pairs']
        
        # 基础格式验证
        if len(solution) != n or any(len(p)!=2 or p[0]==p[1] for p in solution):
            return False
        
        # 构建颜色映射表
        color_map = {}
        for (a, b), (ca, cb) in zip(pairs, solution):
            color_map[a] = ca
            color_map[b] = cb
        
        # 检查完整映射
        if len(color_map) != 2 * n:
            return False
        
        # 验证三连续条件
        for chair in range(1, 2*n + 1):
            next_chair = chair % (2*n) + 1
            next_next_chair = (chair + 1) % (2*n) + 1
            if color_map[chair] == color_map[next_chair] == color_map[next_next_chair]:
                return False
        
        return True
