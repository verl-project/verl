"""# 

### 谜题描述
Pasha and Akim were making a forest map — the lawns were the graph's vertexes and the roads joining the lawns were its edges. They decided to encode the number of laughy mushrooms on every lawn in the following way: on every edge between two lawns they wrote two numbers, the greatest common divisor (GCD) and the least common multiple (LCM) of the number of mushrooms on these lawns. But one day Pasha and Akim had an argument about the laughy mushrooms and tore the map. Pasha was left with just some part of it, containing only m roads. Your task is to help Pasha — use the map he has to restore the number of mushrooms on every lawn. As the result is not necessarily unique, help Pasha to restore any one or report that such arrangement of mushrooms does not exist. It is guaranteed that the numbers on the roads on the initial map were no less that 1 and did not exceed 106.

Input

The first line contains two numbers n and m (<image>) which are the numbers of lawns and roads we know about. Each of the following m lines contains four numbers which are the numbers of lawns the road connects, the GCD and the LCM of the numbers of mushrooms on these lawns (1 ≤ GCD, LCM ≤ 106).

It is guaranteed, that no road connects lawn to itself, and no two lawns are connected by more than one road.

Output

The answer should contain \"YES\" or \"NO\" on the first line, saying whether it is possible or not to perform the arrangement. If the answer is \"YES\", print on the following line n numbers which are the numbers of mushrooms on the corresponding lawns.

Examples

Input

1 0


Output

YES
1 

Input

2 1
1 2 1 3


Output

YES
1 3 

Input

3 2
3 2 1 2
3 1 1 10


Output

YES
5 1 2 

Input

2 1
1 2 3 7


Output

NO

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
from collections import *
from fractions import gcd
nok=lambda x,y: x*y/gcd(x,y)
def fail():
    print \"NO\"
    sys.exit(0)
n,m = map(int,raw_input().split())
r = [map(int,raw_input().split()) for i in xrange(m)]
a0 = [1]*n
aa = [1]*n
for x,y,nd,nk in r:
    a0[x-1]=nok(a0[x-1],nd)
    a0[y-1]=nok(a0[y-1],nd)
rl = [[] for i in range(n)]    
for x,y,nd,nk in r:
    kk= nok(a0[x-1],a0[y-1])    
    if nk%kk != 0: fail()
    kk = nk/kk
    rl[x-1].append((y-1,kk,nd))
    rl[y-1].append((x-1,kk,nd))
f = [False]*n
for i in xrange(n):
    if f[i]: continue
    lk = [k for v,k,kd in rl[i]]
    no = reduce(gcd,lk) if lk else 1
    fl = False
    for k in range(1,no+1):
        if no%k!=0: continue
        a = [i]        
        f[i]=True
        aa[i]=k
        fl = True
        for v in a:        
            vk = aa[v]
            for nv,nk,nd in rl[v]:                    
                if nk%vk != 0:
                    fl=False
                    break
                rk = nk/vk
                if f[nv]:
                    if rk!=aa[nv]: 
                        fl=False
                        break
                else:                    
                    if gcd(a0[nv]*rk,vk*a0[v])>nd:
                        fl=False
                        break                        
                    f[nv]=True
                    aa[nv]=rk
                    a.append(nv)
        if fl: 
            break
        for k in a: f[k]=False
    if not fl: fail()
print \"YES\"
print ' '.join(map(lambda a,b: str(a*b),a0,aa))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from functools import reduce
from collections import deque  # 改用双端队列提高BFS性能

def lcm(a, b):
    return a * b // math.gcd(a, b)

def solve_case(n, m, edges):
    try:
        # 预处理阶段增加输入合法性校验
        for u, v, g, l in edges:
            if g > l or l % g != 0:
                return (False, None)
            if math.gcd(g, l//g) != 1:
                return (False, None)

        a0 = [1] * n
        # 构建a0阶段
        for u, v, g, _ in edges:
            a0[u-1] = lcm(a0[u-1], g)
            a0[v-1] = lcm(a0[v-1], g)

        # 构建关系图
        adjacency = [[] for _ in range(n)]
        for u, v, g, l in edges:
            u_idx, v_idx = u-1, v-1
            k_base = lcm(a0[u_idx], a0[v_idx])
            if l % k_base != 0:
                return (False, None)
            k = l // k_base
            adjacency[u_idx].append((v_idx, k, g))
            adjacency[v_idx].append((u_idx, k, g))

        # 连通分量处理
        solution = [1] * n
        visited = [False] * n
        for i in range(n):
            if visited[i]:
                continue
                
            # BFS遍历连通分量
            q = deque([i])
            visited[i] = True
            divisors = []
            for node in q:
                for _, k, _ in adjacency[node]:
                    divisors.append(k)
            
            # 计算最大公约数
            base_divisor = reduce(math.gcd, divisors, 0) if divisors else 1
            
            # 寻找有效因子
            found = False
            for d in range(1, base_divisor + 1):
                if base_divisor % d != 0:
                    continue
                    
                temp_sol = {i: d}
                valid = True
                bfs_q = deque([i])
                
                while bfs_q and valid:
                    current = bfs_q.popleft()
                    current_d = temp_sol[current]
                    
                    for neighbor, k, g in adjacency[current]:
                        required = k // current_d
                        
                        if neighbor in temp_sol:
                            if temp_sol[neighbor] != required:
                                valid = False
                                break
                            continue
                            
                        if k % current_d != 0:
                            valid = False
                            break
                            
                        # 验证边条件
                        a_val = a0[current] * current_d
                        b_val = a0[neighbor] * required
                        if math.gcd(a_val, b_val) != g or lcm(a_val, b_val) != (a_val * b_val) // g:
                            valid = False
                            break
                            
                        temp_sol[neighbor] = required
                        bfs_q.append(neighbor)
                        
                if valid:
                    for node in temp_sol:
                        solution[node] = temp_sol[node]
                    found = True
                    break
                    
            if not found:
                return (False, None)

        # 最终有效性检查
        final = [a0[i] * solution[i] for i in range(n)]
        for num in final:
            if not (1 <= num <= 10**6):
                return (False, None)
                
        for u, v, g, l in edges:
            a, b = final[u-1], final[v-1]
            if math.gcd(a, b) != g or lcm(a, b) != l:
                return (False, None)
                
        return (True, final)
        
    except:
        return (False, None)

class Cmushroomstrifebootcamp(Basebootcamp):
    def __init__(self, solvable_prob=0.5, min_n=1, max_n=8, min_m=0, max_m=15):
        self.solvable_prob = solvable_prob
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m

    def case_generator(self):
        if random.random() < self.solvable_prob:
            return self._generate_valid_case()
        else:
            return self._generate_invalid_case()

    def _generate_valid_case(self):
        """生成保证有解的案例"""
        while True:
            n = random.randint(self.min_n, self.max_n)
            max_edges = n*(n-1)//2
            m = random.randint(self.min_m, min(self.max_m, max_edges))
            
            # 生成合法顶点值
            nodes = [random.randint(1, 100) for _ in range(n)]  # 小范围便于生成合法边
            edges = []
            
            # 随机选择不重复的边
            existing_edges = set()
            for _ in range(m):
                while True:
                    u = random.randint(1, n)
                    v = random.randint(1, n)
                    if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
                        break
                existing_edges.add((u, v))
                
                a, b = nodes[u-1], nodes[v-1]
                g = math.gcd(a, b)
                l = lcm(a, b)
                edges.append((u, v, g, l))
            
            case = {"n": n, "m": m, "edges": edges}
            # 验证生成的案例确实有解
            possible, _ = solve_case(n, m, edges)
            if possible:
                return case

    def _generate_invalid_case(self):
        """生成保证无解的案例"""
        while True:
            n = random.randint(max(2, self.min_n), self.max_n)  # 至少2个节点才能有边
            m = random.randint(1, min(self.max_m, n*(n-1)//2))
            
            edges = []
            existing_edges = set()
            
            # 强制至少包含一个矛盾边
            invalid_added = False
            for _ in range(m):
                u = random.randint(1, n)
                v = random.randint(1, n)
                while u == v or (u, v) in existing_edges or (v, u) in existing_edges:
                    u = random.randint(1, n)
                    v = random.randint(1, n)
                    
                existing_edges.add((u, v))
                
                if not invalid_added and random.random() < 0.5:
                    # 生成矛盾的gcd和lcm对
                    g = random.randint(2, 50)
                    l = g * random.randint(1, 100) + random.randint(1, g-1)  # 保证l % g != 0
                    invalid_added = True
                else:
                    # 生成合法对
                    a = random.randint(1, 100)
                    b = random.randint(1, 100)
                    g = math.gcd(a, b)
                    l = lcm(a, b)
                    
                edges.append((u, v, g, l))
            
            case = {"n": n, "m": m, "edges": edges}
            possible, _ = solve_case(n, m, edges)
            if not possible:
                return case

    @staticmethod
    def prompt_func(case):
        n = case["n"]
        m = case["m"]
        edges = case["edges"]
        problem = [
            "As a forest map restorer, determine if the mushroom counts can be reconstructed from the given GCD/LCM edges.",
            f"Input:\n{n} {m}",
            "Edge list (u v GCD LCM):"
        ]
        problem.extend(f"{u} {v} {g} {l}" for u, v, g, l in edges)
        problem.append(
            "Output format:\n"
            "First line: YES/NO\n"
            "If YES, second line: space-separated integers\n"
            "Put your final answer within [answer] and [/answer] tags."
        )
        return '\n'.join(problem)

    @staticmethod
    def extract_output(text):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', text, re.DOTALL)
        if not matches:
            return None
            
        answer = matches[-1].strip().upper()
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        
        if not lines:
            return None
            
        if lines[0] == 'NO' and len(lines) == 1:
            return 'NO'
            
        if lines[0] == 'YES' and len(lines) >= 2:
            try:
                numbers = list(map(int, lines[1].split()))
                return numbers
            except:
                pass
        return None

    @classmethod
    def _verify_correction(cls, solution, case):
        # 严格类型检查
        if isinstance(solution, list):
            if len(solution) != case["n"]:
                return False
            try:
                nums = list(map(int, solution))
                if any(not (1 <= x <= 10**6) for x in nums):
                    return False
            except:
                return False
            
            for u, v, g, l in case["edges"]:
                a, b = nums[u-1], nums[v-1]
                actual_gcd = math.gcd(a, b)
                actual_lcm = (a * b) // actual_gcd
                if actual_gcd != g or actual_lcm != l:
                    return False
            return True
            
        elif solution == 'NO':
            possible, _ = solve_case(case["n"], case["m"], case["edges"])
            return not possible
            
        return False
