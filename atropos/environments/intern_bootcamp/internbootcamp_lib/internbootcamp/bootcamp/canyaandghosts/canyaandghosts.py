"""# 

### 谜题描述
Anya loves to watch horror movies. In the best traditions of horror, she will be visited by m ghosts tonight. Anya has lots of candles prepared for the visits, each candle can produce light for exactly t seconds. It takes the girl one second to light one candle. More formally, Anya can spend one second to light one candle, then this candle burns for exactly t seconds and then goes out and can no longer be used.

For each of the m ghosts Anya knows the time at which it comes: the i-th visit will happen wi seconds after midnight, all wi's are distinct. Each visit lasts exactly one second.

What is the minimum number of candles Anya should use so that during each visit, at least r candles are burning? Anya can start to light a candle at any time that is integer number of seconds from midnight, possibly, at the time before midnight. That means, she can start to light a candle integer number of seconds before midnight or integer number of seconds after a midnight, or in other words in any integer moment of time.

Input

The first line contains three integers m, t, r (1 ≤ m, t, r ≤ 300), representing the number of ghosts to visit Anya, the duration of a candle's burning and the minimum number of candles that should burn during each visit. 

The next line contains m space-separated numbers wi (1 ≤ i ≤ m, 1 ≤ wi ≤ 300), the i-th of them repesents at what second after the midnight the i-th ghost will come. All wi's are distinct, they follow in the strictly increasing order.

Output

If it is possible to make at least r candles burn during each visit, then print the minimum number of candles that Anya needs to light for that.

If that is impossible, print  - 1.

Examples

Input

1 8 3
10


Output

3


Input

2 10 1
5 8


Output

1


Input

1 1 3
10


Output

-1

Note

Anya can start lighting a candle in the same second with ghost visit. But this candle isn't counted as burning at this visit.

It takes exactly one second to light up a candle and only after that second this candle is considered burning; it means that if Anya starts lighting candle at moment x, candle is buring from second x + 1 to second x + t inclusively.

In the first sample test three candles are enough. For example, Anya can start lighting them at the 3-rd, 5-th and 7-th seconds after the midnight.

In the second sample test one candle is enough. For example, Anya can start lighting it one second before the midnight.

In the third sample test the answer is  - 1, since during each second at most one candle can burn but Anya needs three candles to light up the room at the moment when the ghost comes.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
m,t,r = map(int,raw_input().split())
a=map(int,raw_input().split())
if r > t: print('-1')
else:
	cand=[]
	ans=0
	for i in a:
		lite=0
		for j in cand:
			dif=i-j
			if dif<=t:
				lite+=1
		to_lite=r-lite
		c=i-1
		for j in xrange(to_lite):
			ans+=1
			cand.append(c)
			c-=1
	print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def solve_case(m, t, r, w_list):
    if r > t:
        return -1
    cand = []
    ans = 0
    for wi in w_list:
        lite = 0
        for c in cand:
            # 蜡烛在wi秒时是否燃烧的条件：c+1 <= wi <= c+t
            if c + 1 <= wi <= c + t:
                lite += 1
        to_lite = r - lite
        if to_lite > 0:
            # 需要从wi-1开始倒推点燃时间
            latest_start = wi - 1
            earliest_start = latest_start - (to_lite - 1)
            if earliest_start < 0:  # 时间不足以点燃所需蜡烛
                return -1
            for start_time in range(latest_start, earliest_start - 1, -1):
                cand.append(start_time)
                ans += 1
    # 最终有效性验证（处理多个幽灵共享蜡烛的情况）
    for wi in w_list:
        burning = 0
        for c in cand:
            if c + 1 <= wi <= c + t:
                burning += 1
        if burning < r:
            return -1
    return ans

class Canyaandghostsbootcamp(Basebootcamp):
    def __init__(self, m_lim=(1, 300), t_lim=(1, 300), r_lim=(1, 300), w_lim=(1, 300)):
        self.m_min, self.m_max = m_lim
        self.t_min, self.t_max = t_lim
        self.r_min, self.r_max = r_lim
        self.w_min, self.w_max = w_lim

    def case_generator(self):
        # 保证有足够的时间点可选
        available_w_slots = self.w_max - self.w_min + 1
        m = random.randint(
            max(self.m_min, 1),
            min(self.m_max, available_w_slots)
        )
        
        t = random.randint(self.t_min, self.t_max)
        r = random.randint(self.r_min, self.r_max)  # 不再限制r<=t
        
        # 生成严格递增的时间序列
        w = sorted(random.sample(range(self.w_min, self.w_max+1), m))
        
        # 确保时间严格递增
        for i in range(1, len(w)):
            if w[i] <= w[i-1]:
                w[i] = w[i-1] + 1
        
        expected = solve_case(m, t, r, w)
        return {
            'm': m,
            't': t,
            'r': r,
            'w': w,
            'expected': expected
        }

    @staticmethod
    def prompt_func(case) -> str:
        params = case
        prompt = f"""Anya今晚将遭遇{params['m']}个鬼魂的拜访。已知：
- 每个蜡烛燃烧时长：{params['t']}秒
- 每个鬼魂来访时需要至少{params['r']}支蜡烛同时燃烧
- 鬼魂来访时间（午夜后的秒数，严格递增）：{', '.join(map(str, params['w']))}

规则细节：
1. 点燃蜡烛需要1秒完整时间（在x秒点燃，x+1秒开始燃烧）
2. 同一秒只能点燃一支蜡烛
3. 蜡烛在x+1到x+{params['t']}秒期间保持燃烧（共{params['t']}秒）
4. 鬼魂来访的瞬间必须有至少{params['r']}支燃烧中的蜡烛

请计算所需的最少蜡烛数量，如果不可能实现，输出-1。答案请用[answer]...[/answer]包裹。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\s*\]([-+]?\d+)\s*\[/answer\s*\]', output, re.IGNORECASE)
        try:
            return int(matches[-1]) if matches else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['expected']
        except:
            return False
