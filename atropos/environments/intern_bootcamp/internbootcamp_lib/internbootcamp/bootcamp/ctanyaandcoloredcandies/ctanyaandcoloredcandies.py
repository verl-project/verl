"""# 

### 谜题描述
There are n candy boxes in front of Tania. The boxes are arranged in a row from left to right, numbered from 1 to n. The i-th box contains r_i candies, candies have the color c_i (the color can take one of three values ​​— red, green, or blue). All candies inside a single box have the same color (and it is equal to c_i).

Initially, Tanya is next to the box number s. Tanya can move to the neighbor box (that is, with a number that differs by one) or eat candies in the current box. Tanya eats candies instantly, but the movement takes one second.

If Tanya eats candies from the box, then the box itself remains in place, but there is no more candies in it. In other words, Tanya always eats all the candies from the box and candies in the boxes are not refilled.

It is known that Tanya cannot eat candies of the same color one after another (that is, the colors of candies in two consecutive boxes from which she eats candies are always different). In addition, Tanya's appetite is constantly growing, so in each next box from which she eats candies, there should be strictly more candies than in the previous one.

Note that for the first box from which Tanya will eat candies, there are no restrictions on the color and number of candies.

Tanya wants to eat at least k candies. What is the minimum number of seconds she will need? Remember that she eats candies instantly, and time is spent only on movements.

Input

The first line contains three integers n, s and k (1 ≤ n ≤ 50, 1 ≤ s ≤ n, 1 ≤ k ≤ 2000) — number of the boxes, initial position of Tanya and lower bound on number of candies to eat. The following line contains n integers r_i (1 ≤ r_i ≤ 50) — numbers of candies in the boxes. The third line contains sequence of n letters 'R', 'G' and 'B', meaning the colors of candies in the correspondent boxes ('R' for red, 'G' for green, 'B' for blue). Recall that each box contains candies of only one color. The third line contains no spaces.

Output

Print minimal number of seconds to eat at least k candies. If solution doesn't exist, print \"-1\".

Examples

Input

5 3 10
1 2 3 4 5
RGBRR


Output

4


Input

2 1 15
5 6
RG


Output

-1

Note

The sequence of actions of Tanya for the first example:

  * move from the box 3 to the box 2; 
  * eat candies from the box 2; 
  * move from the box 2 to the box 3; 
  * eat candy from the box 3; 
  * move from the box 3 to the box 4; 
  * move from the box 4 to the box 5; 
  * eat candies from the box 5. 



Since Tanya eats candy instantly, the required time is four seconds.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, s, k=map(int, raw_input().split())
s-=1
r=map(int, raw_input().split())
color=raw_input()
dp=[[] for i in range(n)]
def dfs(cur):
	if dp[cur]:
		return
	dp[cur]=[0]*(r[cur]+1)+[1000000000]*(k-r[cur])
	for to in range(n):
		if color[to]!=color[cur] and r[to]>r[cur]:
			dfs(to)
			dis=abs(cur-to)
			for i in range(r[cur]+1, k+1):
				dp[cur][i]=min(dp[cur][i], dp[to][i-r[cur]]+dis)
result=1000000000
for i in range(n):
	dfs(i)
	result=min(result, abs(i-s)+dp[i][k])
print result if result!=1000000000 else -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def solve_candy_boxes(n, s, k, r_list, color_str):
    s -= 1  # 转换为0-based索引
    r = r_list
    color = color_str
    INF = float('inf')
    
    # 预处理最大可能的糖果数
    max_possible = sum(r)
    if max_possible < k:
        return -1
    
    # 动态规划数组，dp[cur][c]表示从cur出发，获得至少c颗糖果的最短时间
    dp = [[INF] * (k + 1) for _ in range(n)]
    
    # 预处理每个盒子自身的情况
    for i in range(n):
        current_max = min(r[i], k)
        for c in range(current_max + 1):
            dp[i][c] = 0  # 只需要吃当前盒子即可
        
    # 记忆化搜索函数
    def dfs(cur):
        # 已经处理过的情况直接返回
        if dp[cur][k] != INF:
            return
        
        # 尝试所有可能的后继盒子
        for to in range(n):
            if color[to] != color[cur] and r[to] > r[cur]:
                dfs(to)
                distance = abs(cur - to)
                
                # 状态转移：当前吃掉的糖果数 + 后续吃掉的糖果数
                for c in range(k, -1, -1):
                    if dp[cur][c] == INF:
                        continue
                    
                    # 计算转移后的糖果数
                    new_c = min(c + r[to], k)
                    cost = dp[cur][c] + distance
                    if cost < dp[to][new_c]:
                        dp[to][new_c] = cost
                        # 回溯更新所有可能的更优解
                        for nc in range(new_c, k+1):
                            if dp[to][nc] > cost:
                                dp[to][nc] = cost
    
    # 从每个可能的起点开始计算
    for i in range(n):
        dfs(i)
    
    # 计算最小时间
    min_time = INF
    for i in range(n):
        start_cost = abs(i - s)
        if start_cost + dp[i][k] < min_time:
            min_time = start_cost + dp[i][k]
    
    return min_time if min_time != INF else -1

class Ctanyaandcoloredcandiesbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=15, min_r=1, max_r=20, min_k=5, max_k=200):
        self.min_n = min_n
        self.max_n = max_n
        self.min_r = min_r
        self.max_r = max_r
        self.min_k = min_k
        self.max_k = max_k
    
    def case_generator(self):
        while True:
            n = random.randint(self.min_n, self.max_n)
            s = random.randint(1, n)
            r = [random.randint(self.min_r, self.max_r) for _ in range(n)]
            colors = ''.join(random.choice(['R', 'G', 'B']) for _ in range(n))
            total = sum(r)
            k = random.randint(self.min_k, min(total + 5, self.max_k))
            
            # 确保至少存在两种颜色的盒子
            if len(set(colors)) >= 2:
                return {
                    'n': n,
                    's': s,
                    'k': k,
                    'r': r,
                    'colors': colors
                }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = question_case['s']
        k = question_case['k']
        r = question_case['r']
        colors = question_case['colors']
        table = "\n| 盒子编号 | 糖果数量 | 颜色 |\n"
        table += "|:-:|:-:|:-:|\n"
        for i in range(n):
            table += f"| {i+1} | {r[i]} | {colors[i]} |\n"
        prompt = f"""## 糖果盒谜题

你面前有{n}个排列成行的糖果盒（编号1~{n}），初始位置在盒子{s}旁。每个盒子的信息如下：

{table}

### 规则说明
1. 每次可以移动到相邻盒子（耗时1秒）或吃光当前盒子的所有糖果（瞬间完成）
2. 连续吃的两个盒子颜色必须不同
3. 后吃的盒子糖果数必须严格大于前一个
4. 目标是通过移动和吃糖获得**至少{k}颗糖果**

请计算达成目标所需的最短时间（单位：秒），如果无法达成，请输出-1。

将最终答案放在[answer]和[/answer]之间，例如：[answer]5[/answer]或[answer]-1[/answer]。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 支持多种格式的答案提取，包含可能的换行和空格
        matches = re.findall(r'\[answer\s*\]\s*(-?\d+)\s*\[/answer\s*\]', output, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1].strip())
            except:
                return None
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 直接调用求解函数验证答案正确性
            ground_truth = solve_candy_boxes(
                identity['n'],
                identity['s'],
                identity['k'],
                identity['r'],
                identity['colors']
            )
            return solution == ground_truth
        except:
            return False
