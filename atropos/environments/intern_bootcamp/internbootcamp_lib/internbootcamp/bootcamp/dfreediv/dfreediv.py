"""# 

### 谜题描述
Vasya plays FreeDiv. In this game he manages a huge state, which has n cities and m two-way roads between them. Unfortunately, not from every city you can reach any other one moving along these roads. Therefore Vasya decided to divide the state into provinces so that in every province, one could reach from every city all the cities of the province, but there are no roads between provinces. 

Unlike other turn-based strategies, in FreeDiv a player has the opportunity to build tunnels between cities. The tunnels are two-way roads along which one can move armies undetected by the enemy. However, no more than one tunnel can be connected to each city. As for Vasya, he wants to build a network of tunnels so that any pair of cities in his state were reachable by some path consisting of roads and a tunnels. But at that no more than k tunnels are connected to each province (otherwise, the province will be difficult to keep in case other provinces are captured by enemy armies).

Vasya discovered that maybe he will not be able to build such a network for the current condition of the state. Maybe he'll have first to build several roads between cities in different provinces to merge the provinces. Your task is to determine the minimum number of roads Vasya needs to build so that it was possible to build the required network of tunnels in the resulting state.

Input

The first line contains three integers n, m and k (1 ≤ n, k ≤ 106, 0 ≤ m ≤ 106). Each of the next m lines contains two integers. They are the numbers of cities connected by a corresponding road. No road connects city to itself and there is at most one road between each pair of cities.

Output

Print a single number, the minimum number of additional roads.

Examples

Input

3 3 2
1 2
2 3
3 1


Output

0

Input

4 2 2
1 2
3 4


Output

0

Input

4 0 2


Output

1

Note

In the first example only one province exists, so it is not necessary to build any tunnels or roads.

In the second example two provinces exist. It is possible to merge the provinces by building a tunnel between cities 1 and 3.

In the third example at least one additional road is necessary. For example it is possible to build additional road between cities 1 and 2 and build two tunnels between cities 1 and 3, 2 and 4 after that.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
rl=sys.stdin.readline
n,m,k=map(int,rl().split())
c = [-1]*n
def root(x):
    p = x
    while c[p]>=0: p=c[p]
    while c[x]>=0: 
        t = c[x]
        c[x] = p
        x = t
    return p
#def root(x):
#    if c[x]<0: return x
#    c[x]=root(c[x])
#    return c[x]
for i in xrange(m):
    x,y=rl().split()
    f = root(int(x)-1)
    t = root(int(y)-1)
    if f==t: continue
    if (c[f]&1): f,t=t,f
    c[t]+=c[f]
    c[f]=t
l,s,q=0,2,0
for i in xrange(n):
    if c[i]>=0: continue
    j=k if c[i]<-k else -c[i]
    if j==1: l+=1
    else: s+=j-2
    q+=1
    
if l==1: print 0
elif k==1: print q-2;
elif l<=s: print 0;
else: print (l-s+1)/2;
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Dfreedivbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_k=5, max_m=20):
        self.max_n = max_n
        self.max_k = max_k
        self.max_m = max_m
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        k = random.randint(1, self.max_k)
        
        # 生成无重复道路
        available = set()
        for i in range(1, n+1):
            for j in range(i+1, n+1):
                available.add((i,j))
        available = list(available)
        random.shuffle(available)
        m = min(len(available), random.randint(0, self.max_m))
        roads = available[:m]
        
        return {
            'n': n,
            'm': m,
            'k': k,
            'roads': roads,
            'correct_answer': self._compute_answer(n, k, roads)
        }
    
    @staticmethod
    def prompt_func(question_case):
        return (
            f"Input:\n{question_case['n']} {question_case['m']} {question_case['k']}\n" +
            '\n'.join(f"{x} {y}" for x,y in question_case['roads']) +
            "\nOutput the minimal additional roads required within [answer][/answer]."
        )
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None
    
    @classmethod
    def _verify_correction(cls, sol, case):
        return sol == case['correct_answer']
    
    @staticmethod
    def _compute_answer(n, k, roads):
        """完全实现原题算法逻辑"""
        parent = list(range(n))
        size = [1]*n
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # 路径压缩
                u = parent[u]
            return u
        
        # 合并初始道路
        for x,y in roads:
            u = find(int(x)-1)
            v = find(int(y)-1)
            if u == v: continue
            if size[u] < size[v]:  # 小树合并到大树
                parent[u] = v
                size[v] += size[u]
            else:
                parent[v] = u
                size[u] += size[v]
        
        # 统计连通分量
        roots = defaultdict(int)
        for i in range(n):
            roots[find(i)] += 1
        components = list(roots.values())
        
        # 原题核心逻辑
        l = s = q = 0
        for cnt in components:
            j = min(cnt, k)
            if j == 1:
                l += 1
            else:
                s += j - 2
            q += 1
        
        if q == 1:
            return 0
        if k == 1:
            return max(q - 2, 0)
        if l <= s:
            return 0
        return (l - s + 1) // 2
