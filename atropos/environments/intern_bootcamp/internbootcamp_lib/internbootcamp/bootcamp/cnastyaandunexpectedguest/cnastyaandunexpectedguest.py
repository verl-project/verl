"""# 

### 谜题描述
If the girl doesn't go to Denis, then Denis will go to the girl. Using this rule, the young man left home, bought flowers and went to Nastya. 

On the way from Denis's house to the girl's house is a road of n lines. This road can't be always crossed in one green light. Foreseeing this, the good mayor decided to place safety islands in some parts of the road. Each safety island is located after a line, as well as at the beginning and at the end of the road. Pedestrians can relax on them, gain strength and wait for a green light.

Denis came to the edge of the road exactly at the moment when the green light turned on. The boy knows that the traffic light first lights up g seconds green, and then r seconds red, then again g seconds green and so on.

Formally, the road can be represented as a segment [0, n]. Initially, Denis is at point 0. His task is to get to point n in the shortest possible time.

He knows many different integers d_1, d_2, …, d_m, where 0 ≤ d_i ≤ n — are the coordinates of points, in which the safety islands are located. Only at one of these points, the boy can be at a time when the red light is on.

Unfortunately, Denis isn't always able to control himself because of the excitement, so some restrictions are imposed:

  * He must always move while the green light is on because it's difficult to stand when so beautiful girl is waiting for you. Denis can change his position by ± 1 in 1 second. While doing so, he must always stay inside the segment [0, n]. 
  * He can change his direction only on the safety islands (because it is safe). This means that if in the previous second the boy changed his position by +1 and he walked on a safety island, then he can change his position by ± 1. Otherwise, he can change his position only by +1. Similarly, if in the previous second he changed his position by -1, on a safety island he can change position by ± 1, and at any other point by -1. 
  * At the moment when the red light is on, the boy must be on one of the safety islands. He can continue moving in any direction when the green light is on. 



Denis has crossed the road as soon as his coordinate becomes equal to n.

This task was not so simple, because it's possible that it is impossible to cross the road. Since Denis has all thoughts about his love, he couldn't solve this problem and asked us to help him. Find the minimal possible time for which he can cross the road according to these rules, or find that it is impossible to do.

Input

The first line contains two integers n and m (1 ≤ n ≤ 10^6, 2 ≤ m ≤ min(n + 1, 10^4)) — road width and the number of safety islands.

The second line contains m distinct integers d_1, d_2, …, d_m (0 ≤ d_i ≤ n) — the points where the safety islands are located. It is guaranteed that there are 0 and n among them.

The third line contains two integers g, r (1 ≤ g, r ≤ 1000) — the time that the green light stays on and the time that the red light stays on.

Output

Output a single integer — the minimum time for which Denis can cross the road with obeying all the rules.

If it is impossible to cross the road output -1.

Examples

Input


15 5
0 3 7 14 15
11 11


Output


45

Input


13 4
0 3 7 13
9 9


Output


-1

Note

In the first test, the optimal route is: 

  * for the first green light, go to 7 and return to 3. In this case, we will change the direction of movement at the point 7, which is allowed, since there is a safety island at this point. In the end, we will be at the point of 3, where there is also a safety island. The next 11 seconds we have to wait for the red light. 
  * for the second green light reaches 14. Wait for the red light again. 
  * for 1 second go to 15. As a result, Denis is at the end of the road. 



In total, 45 seconds are obtained.

In the second test, it is impossible to cross the road according to all the rules.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

_,n = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
A.sort()
g,r = [int(x) for x in input().split()]

D = []
for i in range(n - 1):
    D.append(A[i + 1] - A[i])

found = [0]*((g + 1)*n)

count = -1
finder = False
next_layer = [0]
while next_layer and not finder:
    count += 1
    bfs = next_layer
    next_layer = []
    for node in bfs:
        t,i = divmod(node, n)
        if i > 0:
            tp = t + D[i - 1]
            ind = i - 1 + n * tp
            if tp <= g and not found[ind]:
                found[ind] = 1
                if tp == g:
                    next_layer.append(i - 1)
                else:
                    bfs.append(ind)
        if i + 1 < n:
            tp = t + D[i]
            ind = i + 1 + n * tp
            if tp <= g and not found[ind]:
                found[ind] = 1
                if i + 1 == n - 1:
                    finder = True
                elif tp == g:
                    next_layer.append(i + 1)
                else:
                    bfs.append(ind)

if finder:
    print count*(g+r) + min(i for i in range(1, g + 1) if found[n - 1 + i * n])
else:
    print -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
from collections import deque

def compute_min_time(d_sorted, g, r):
    m = len(d_sorted)
    if m < 2 or d_sorted[-1] != d_sorted[0] + sum(d_sorted[i+1]-d_sorted[i] for i in range(m-1)):
        return -1  # Invalid safety island configuration
    
    D = [d_sorted[i+1] - d_sorted[i] for i in range(m-1)]
    n = m  # 安全岛数量作为参考代码中的n
    
    # BFS状态初始化
    found = [0] * ((g + 1) * n)
    count = -1
    finder = False
    next_layer = [0]  # 初始状态：在第一个安全岛，剩余时间g
    
    while next_layer and not finder:
        count += 1
        current_layer = next_layer
        next_layer = []
        for node in current_layer:
            t, i = divmod(node, n)  # t: 已用时间, i: 当前安全岛索引
            
            # 向左移动
            if i > 0:
                delta = D[i-1]
                new_t = t + delta
                if new_t <= g:
                    new_i = i - 1
                    index = new_i + n * new_t
                    if not found[index]:
                        found[index] = 1
                        if new_i == n-1:  # 到达最后一个安全岛
                            finder = True
                        if new_t == g:     # 当前周期时间耗尽
                            next_layer.append(new_i)
                        else:             # 同一周期继续移动
                            current_layer.append(index)
            
            # 向右移动 
            if i < n-1:
                delta = D[i]
                new_t = t + delta
                if new_t <= g:
                    new_i = i + 1
                    index = new_i + n * new_t
                    if not found[index]:
                        found[index] = 1
                        if new_i == n-1:
                            finder = True
                        if new_t == g:
                            next_layer.append(new_i)
                        else:
                            current_layer.append(index)
    
    if not finder:
        return -1
    
    # 计算最小时间
    min_time = float('inf')
    for t in range(1, g+1):
        if found[(n-1) + n*t]:
            total = count*(g+r) + t
            min_time = min(min_time, total)
    
    return min_time if min_time != float('inf') else -1

class Cnastyaandunexpectedguestbootcamp(Basebootcamp):
    def __init__(self, **params):
        # 参数范围优化
        self.max_n = params.get('max_n', 10**6)
        self.max_m = params.get('max_m', 10**4)
        self.max_gr = params.get('max_gr', 1000)
    
    def case_generator(self):
        # 生成参数时保持数学合理性
        n = random.randint(1, min(self.max_n, 10**4))  # 控制过大数据生成
        m = random.randint(2, min(self.max_m, n+1))
        
        # 生成安全岛位置
        d = [0, n]
        if m > 2:
            candidates = sorted(random.sample(range(1, n), k=min(m-2, n-1)))
            d.extend(candidates)
            d = sorted(list(set(d)))  # 去重并排序
            while len(d) < m:  # 补充不足的位置
                new_p = random.randint(1, n-1)
                if new_p not in d:
                    d.append(new_p)
                d = sorted(d)
        
        # 确保包含0和n
        d = sorted(list(set(d).union({0, n})))
        m = len(d)
        g = random.randint(1, self.max_gr)
        r = random.randint(1, self.max_gr)
        
        return {
            'n': max(d),  # 实际道路长度
            'm': m,
            'd': d,
            'g': g,
            'r': r
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        d_str = ' '.join(map(str, sorted(question_case['d'])))
        g = question_case['g']
        r = question_case['r']
        prompt = f"""# 道路穿越问题

## 问题描述
Denis需要穿过一条宽{n}米的道路，道路上有{m}个安全岛，位置分别为：{d_str}。交通信号灯周期为绿灯{g}秒，红灯{r}秒。

## 移动规则
1. 只能在绿灯期间移动，每秒移动±1米
2. 方向变更只能在安全岛上进行（包括起点和终点）
3. 红灯期间必须位于安全岛
4. 抵达终点{n}时立即停止

## 计算要求
请计算Denis到达终点的最短时间（秒），若不可能则输出-1。

请将最终答案用[answer]和[/answer]标签包裹，例如：
[answer]45[/answer] 或 [answer]-1[/answer]"""

        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            ans = int(solution)
        except:
            return False
        # 调用修正后的计算函数
        d_sorted = sorted(identity['d'])
        correct = compute_min_time(d_sorted, identity['g'], identity['r'])
        return ans == correct
