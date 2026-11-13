"""# 

### 谜题描述
ZS the Coder and Chris the Baboon has arrived at Udayland! They walked in the park where n trees grow. They decided to be naughty and color the trees in the park. The trees are numbered with integers from 1 to n from left to right.

Initially, tree i has color ci. ZS the Coder and Chris the Baboon recognizes only m different colors, so 0 ≤ ci ≤ m, where ci = 0 means that tree i is uncolored.

ZS the Coder and Chris the Baboon decides to color only the uncolored trees, i.e. the trees with ci = 0. They can color each of them them in any of the m colors from 1 to m. Coloring the i-th tree with color j requires exactly pi, j litres of paint.

The two friends define the beauty of a coloring of the trees as the minimum number of contiguous groups (each group contains some subsegment of trees) you can split all the n trees into so that each group contains trees of the same color. For example, if the colors of the trees from left to right are 2, 1, 1, 1, 3, 2, 2, 3, 1, 3, the beauty of the coloring is 7, since we can partition the trees into 7 contiguous groups of the same color : {2}, {1, 1, 1}, {3}, {2, 2}, {3}, {1}, {3}. 

ZS the Coder and Chris the Baboon wants to color all uncolored trees so that the beauty of the coloring is exactly k. They need your help to determine the minimum amount of paint (in litres) needed to finish the job.

Please note that the friends can't color the trees that are already colored.

Input

The first line contains three integers, n, m and k (1 ≤ k ≤ n ≤ 100, 1 ≤ m ≤ 100) — the number of trees, number of colors and beauty of the resulting coloring respectively.

The second line contains n integers c1, c2, ..., cn (0 ≤ ci ≤ m), the initial colors of the trees. ci equals to 0 if the tree number i is uncolored, otherwise the i-th tree has color ci.

Then n lines follow. Each of them contains m integers. The j-th number on the i-th of them line denotes pi, j (1 ≤ pi, j ≤ 109) — the amount of litres the friends need to color i-th tree with color j. pi, j's are specified even for the initially colored trees, but such trees still can't be colored.

Output

Print a single integer, the minimum amount of paint needed to color the trees. If there are no valid tree colorings of beauty k, print  - 1.

Examples

Input

3 2 2
0 0 0
1 2
3 4
5 6


Output

10

Input

3 2 2
2 1 2
1 3
2 4
3 5


Output

-1

Input

3 2 2
2 0 0
1 3
2 4
3 5


Output

5

Input

3 2 3
2 1 2
1 3
2 4
3 5


Output

0

Note

In the first sample case, coloring the trees with colors 2, 1, 1 minimizes the amount of paint used, which equals to 2 + 3 + 5 = 10. Note that 1, 1, 1 would not be valid because the beauty of such coloring equals to 1 ({1, 1, 1} is a way to group the trees into a single group of the same color).

In the second sample case, all the trees are colored, but the beauty of the coloring is 3, so there is no valid coloring, and the answer is  - 1.

In the last sample case, all the trees are colored and the beauty of the coloring matches k, so no paint is used and the answer is 0. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m, ink = map(int, raw_input().split(' '))
c = map(int, raw_input().split(' '))
p = [map(int, raw_input().split(' ')) for i in xrange(n)]
inf = 10 ** 18
def getminline(line):
    a = [inf]
    for i in line[1:]:
        a.append(min(a[-1], i))
    return a

def getmin(dp):
    dpmin1 = [getminline(i) for i in dp]
    dpmin2 = [getminline(i[::-1])[::-1] for i in dp]
    return dpmin1, dpmin2

dp1 = [[inf if j > 0 else 0 for i in xrange(m + 2)] for j in xrange(n + 1)]
if m == 1:
    dp1[1][1] = 0
dpmin1, dpmin2 = getmin(dp1)
for k in xrange(n):
    dp2 = [[inf for i in xrange(m + 2)] for j in xrange(n + 1)]
    for color in xrange(1, m + 1):
        if c[k] == 0 or c[k] == color:
            for dff in xrange(1, n + 1):
                dp2[dff][color] = min(dp1[dff][color], min(dpmin1[dff - 1][color - 1], dpmin2[dff - 1][color + 1]))
                if c[k] == 0:
                    dp2[dff][color] += p[k][color - 1]

    dp1 = dp2
    #print dp1
    dpmin1, dpmin2 = getmin(dp1)
ans = min(dp1[ink][i] for i in xrange(1, m + 1))
if ans >= inf:
    print -1
else:
    print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
import re
from bootcamp import Basebootcamp

def solve(n, m, k, c_list, p_matrix):
    INF = 10**18
    c = c_list
    p = p_matrix

    # DP优化算法实现
    dp = [[INF]*(m+1) for _ in range(k+1)]
    dp[0][0] = 0  # 初始状态
    
    for tree_idx in range(n):
        current_color = c[tree_idx]
        new_dp = [[INF]*(m+1) for _ in range(k+1)]
        
        for groups in range(k+1):
            for prev_color in range(m+1):
                if dp[groups][prev_color] == INF:
                    continue
                
                for new_color in range(1, m+1):
                    if current_color != 0 and current_color != new_color:
                        continue  # 已染色树不能改变颜色
                    
                    # 计算新分组数
                    new_groups = groups + (1 if new_color != prev_color else 0)
                    if new_groups > k:
                        continue
                    
                    # 计算成本
                    cost = p[tree_idx][new_color-1] if current_color == 0 else 0
                    
                    new_dp[new_groups][new_color] = min(
                        new_dp[new_groups][new_color],
                        dp[groups][prev_color] + cost
                    )
        
        dp = new_dp

    min_cost = min(dp[k][color] for color in range(1, m+1))
    return min_cost if min_cost < INF else -1

class Ccoloringtreesbootcamp(Basebootcamp):
    def __init__(self, max_n=50, max_m=20, **kwargs):
        self.max_n = max_n
        self.max_m = max_m
        
    def case_generator(self):
        # 生成有效案例逻辑优化
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        k = random.randint(1, min(n, 20))  # 限制k的范围
        
        # 生成颜色，确保至少有一个未染色树（当需要染色时）
        c = []
        for _ in range(n):
            if random.random() < 0.3:
                c.append(0)
            else:
                c.append(random.randint(1, m))
        
        # 生成油漆成本矩阵
        p = [[random.randint(1, 1000) for _ in range(m)] for _ in range(n)]
        
        # 计算正确答案
        ans = solve(n, m, k, c, p)
        return {
            "n": n,
            "m": m,
            "k": k,
            "c": c,
            "p": p,
            "ans": ans
        }

    @staticmethod
    def prompt_func(case):
        # 修正的字符串格式化方法
        prompt = f"""ZS the Coder和Chris the Baboon需要给公园里的树染色。现有{case['n']}棵树排成一列，每棵树初始颜色为0（未染色）或1~{case['m']}。要求将所有未染色的树染色，使得最终染色方案的美丽值恰好为{case['k']}（美丽值定义为将树划分为连续同色组的最小数量），求最小油漆用量。

输入格式：
- 第一行三个整数n m k
- 第二行n个整数表示初始颜色
- 接下来n行每行m个整数表示染色花费

当前问题：
第一行：{case['n']} {case['m']} {case['k']}
第二行：{' '.join(map(str, case['c']))}
"""
        # 添加油漆成本矩阵
        prompt += "\n" + "\n".join(' '.join(map(str, row)) for row in case['p'])
        prompt += "\n请计算最小油漆用量（若无解输出-1），将最终答案放在[answer]标签内，例如：[answer]-1[/answer]"
        return prompt

    @staticmethod
    def extract_output(output):
        # 增强的正则匹配模式
        pattern = r'\[answer\][\s\n]*(-?\d+)[\s\n]*\[/answer\]'
        matches = re.findall(pattern, output, re.IGNORECASE|re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 严格验证答案
        return solution == identity['ans']
