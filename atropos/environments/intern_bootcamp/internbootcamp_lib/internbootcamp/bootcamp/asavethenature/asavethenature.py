"""# 

### 谜题描述
You are an environmental activist at heart but the reality is harsh and you are just a cashier in a cinema. But you can still do something!

You have n tickets to sell. The price of the i-th ticket is p_i. As a teller, you have a possibility to select the order in which the tickets will be sold (i.e. a permutation of the tickets). You know that the cinema participates in two ecological restoration programs applying them to the order you chose:

  * The x\% of the price of each the a-th sold ticket (a-th, 2a-th, 3a-th and so on) in the order you chose is aimed for research and spreading of renewable energy sources. 
  * The y\% of the price of each the b-th sold ticket (b-th, 2b-th, 3b-th and so on) in the order you chose is aimed for pollution abatement. 



If the ticket is in both programs then the (x + y) \% are used for environmental activities. Also, it's known that all prices are multiples of 100, so there is no need in any rounding.

For example, if you'd like to sell tickets with prices [400, 100, 300, 200] and the cinema pays 10\% of each 2-nd sold ticket and 20\% of each 3-rd sold ticket, then arranging them in order [100, 200, 300, 400] will lead to contribution equal to 100 ⋅ 0 + 200 ⋅ 0.1 + 300 ⋅ 0.2 + 400 ⋅ 0.1 = 120. But arranging them in order [100, 300, 400, 200] will lead to 100 ⋅ 0 + 300 ⋅ 0.1 + 400 ⋅ 0.2 + 200 ⋅ 0.1 = 130.

Nature can't wait, so you decided to change the order of tickets in such a way, so that the total contribution to programs will reach at least k in minimum number of sold tickets. Or say that it's impossible to do so. In other words, find the minimum number of tickets which are needed to be sold in order to earn at least k.

Input

The first line contains a single integer q (1 ≤ q ≤ 100) — the number of independent queries. Each query consists of 5 lines.

The first line of each query contains a single integer n (1 ≤ n ≤ 2 ⋅ 10^5) — the number of tickets.

The second line contains n integers p_1, p_2, ..., p_n (100 ≤ p_i ≤ 10^9, p_i mod 100 = 0) — the corresponding prices of tickets.

The third line contains two integers x and a (1 ≤ x ≤ 100, x + y ≤ 100, 1 ≤ a ≤ n) — the parameters of the first program.

The fourth line contains two integers y and b (1 ≤ y ≤ 100, x + y ≤ 100, 1 ≤ b ≤ n) — the parameters of the second program.

The fifth line contains single integer k (1 ≤ k ≤ 10^{14}) — the required total contribution.

It's guaranteed that the total number of tickets per test doesn't exceed 2 ⋅ 10^5.

Output

Print q integers — one per query. 

For each query, print the minimum number of tickets you need to sell to make the total ecological contribution of at least k if you can sell tickets in any order.

If the total contribution can not be achieved selling all the tickets, print -1.

Example

Input


4
1
100
50 1
49 1
100
8
100 200 100 200 100 200 100 100
10 2
15 3
107
3
1000000000 1000000000 1000000000
50 1
50 1
3000000000
5
200 100 100 100 100
69 5
31 2
90


Output


-1
6
3
4

Note

In the first query the total contribution is equal to 50 + 49 = 99 < 100, so it's impossible to gather enough money.

In the second query you can rearrange tickets in a following way: [100, 100, 200, 200, 100, 200, 100, 100] and the total contribution from the first 6 tickets is equal to 100 ⋅ 0 + 100 ⋅ 0.1 + 200 ⋅ 0.15 + 200 ⋅ 0.1 + 100 ⋅ 0 + 200 ⋅ 0.25 = 10 + 30 + 20 + 50 = 110.

In the third query the full price of each ticket goes to the environmental activities.

In the fourth query you can rearrange tickets as [100, 200, 100, 100, 100] and the total contribution from the first 4 tickets is 100 ⋅ 0 + 200 ⋅ 0.31 + 100 ⋅ 0 + 100 ⋅ 0.31 = 62 + 31 = 93.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
Q = int(raw_input())
 
def gcd(a,b):
	if not b:
		return a
	if not a:
		return b
	return gcd(b,a%b)
 
for _ in range(Q):
	N = int(raw_input())
	arr = list(map(int,raw_input().split()))
	x,a = map(int,raw_input().split())
	y,b = map(int,raw_input().split())
	if y > x:
		x,y = y,x
		a,b = b,a
	K = int(raw_input())
	arr = sorted(arr,reverse=True)
	li1 = []
	li2 = []
	cur = 0
	lo = 0
	hi = N+1
	while lo < hi:
		mid = (lo+hi)/2
		cnt1 = mid/(a*b/gcd(a,b))
		cnt2 = mid/a-cnt1
		cnt3 = mid/b-cnt1
		ind = 0
		ans = 0
		for i in range(cnt1):
			ans += arr[ind]/100*(x+y)
			ind += 1
		for i in range(cnt2):
			ans += arr[ind]/100*x
			ind += 1
		for i in range(cnt3):
			ans += arr[ind]/100*y
			ind += 1
		if ans >= K:
			hi = mid
		else:
			lo = mid+1
	if lo > N:
		print -1
	else:
		print lo
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

def solve(n, p_list, x, a, y, b, k):
    arr = sorted(p_list, reverse=True)
    # 确保x是较大值并交换参数
    if y > x:
        x, y = y, x
        a, b = b, a
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    g = gcd(a, b)
    lcm_ab = (a * b) // g if g != 0 else 0
    lo = 0
    hi = n
    
    while lo < hi:
        mid = (lo + hi) // 2
        cnt1 = mid // lcm_ab if lcm_ab != 0 else 0
        cnt2 = mid // a - cnt1
        cnt3 = mid // b - cnt1
        
        total = 0
        ind = 0
        # 处理x+y%的贡献
        for _ in range(cnt1):
            if ind >= len(arr):
                break
            total += arr[ind] // 100 * (x + y)
            ind += 1
        # 处理x%的贡献
        for _ in range(cnt2):
            if ind >= len(arr):
                break
            total += arr[ind] // 100 * x
            ind += 1
        # 处理y%的贡献
        for _ in range(cnt3):
            if ind >= len(arr):
                break
            total += arr[ind] // 100 * y
            ind += 1
        
        if total >= k:
            hi = mid
        else:
            lo = mid + 1
    
    return lo if lo <= n else -1  # 移除多余验证

class Asavethenaturebootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=20, p_min=1, p_max=1000, k_gen_strategy='mixed'):
        self.n_min = n_min
        self.n_max = n_max
        self.p_min = p_min  # 100的倍數基數
        self.p_max = p_max
        self.k_gen_strategy = k_gen_strategy
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        p = [random.randint(self.p_min, self.p_max) * 100 for _ in range(n)]
        
        # 生成合法的x,y參數
        s = random.randint(2, 100)
        x = random.randint(1, s-1)
        y = s - x
        
        # 修正：允許a,b在完整範圍隨機
        a = random.randint(1, n)
        b = random.randint(1, n)
        
        # 重要：保持與solve函數相同的參數交換邏輯
        if y > x:
            x, y = y, x
            a, b = b, a
        
        # 計算真實最大值
        sorted_p = sorted(p, reverse=True)
        g = math.gcd(a, b)
        lcm_ab = (a * b) // g if g != 0 else 0
        
        max_total = 0
        ind = 0
        # 計算使用全部票的最大貢獻
        for cnt in [n//lcm_ab, (n//a)-n//lcm_ab, (n//b)-n//lcm_ab]:
            take = min(cnt, len(sorted_p)-ind)
            if cnt == n//lcm_ab:
                rate = x + y
            elif cnt == (n//a)-n//lcm_ab:
                rate = x
            else:
                rate = y
            max_total += sum(sorted_p[ind:ind+take]) // 100 * rate
            ind += take
        
        # 生成k時考慮策略
        if self.k_gen_strategy == 'mixed':
            base = max(1, max_total)
            k = random.randint(1, base * 2)
        elif self.k_gen_strategy == 'solvable':
            k = random.randint(1, max_total) if max_total > 0 else 1
        elif self.k_gen_strategy == 'unsolvable':
            k = max_total + random.randint(1, 1000)
        else:
            raise ValueError("Invalid strategy")
        
        return {
            'n': n,
            'p': p,
            'x': x,
            'a': a,
            'y': y,
            'b': b,
            'k': k,
        }
    
    @staticmethod
    def prompt_func(question_case):
        params = question_case
        problem_desc = (
            "作为电影院售票员兼环保主义者，你需要优化票券销售顺序以达到环保筹款目标。\n\n"
            "**参数说明**\n"
            f"- 可售票据：{params['n']} 张，价格分别为（单位元）{params['p']}\n"
            f"- 环保项目1：每售出第 {params['a']}、{2*params['a']}... 张票时，贡献票价的 {params['x']}%\n"
            f"- 环保项目2：每售出第 {params['b']}、{2*params['b']}... 张票时，贡献票价的 {params['y']}%\n"
            "**叠加规则**：若同一张票同时符合两个项目（如第 {lcm} 张），则贡献率叠加".format(
                lcm=(params['a']*params['b']//math.gcd(params['a'], params['b']))
            ) + "\n\n"
            f"**目标**：通过调整售票顺序，使得用最少的售票数量达到至少 {params['k']} 元的环保捐款。\n"
            "**输出**：最少需要的售票数（如无法达到则返回 -1），答案请包裹在[answer][/answer]标签中。"
        )
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = solve(
                identity['n'], identity['p'],
                identity['x'], identity['a'],
                identity['y'], identity['b'],
                identity['k']
            )
            return solution == expected
        except:
            return False
