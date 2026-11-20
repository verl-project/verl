"""# 

### 谜题描述
After defeating a Blacklist Rival, you get a chance to draw 1 reward slip out of x hidden valid slips. Initially, x=3 and these hidden valid slips are Cash Slip, Impound Strike Release Marker and Pink Slip of Rival's Car. Initially, the probability of drawing these in a random guess are c, m, and p, respectively. There is also a volatility factor v. You can play any number of Rival Races as long as you don't draw a Pink Slip. Assume that you win each race and get a chance to draw a reward slip. In each draw, you draw one of the x valid items with their respective probabilities. Suppose you draw a particular item and its probability of drawing before the draw was a. Then,

  * If the item was a Pink Slip, the quest is over, and you will not play any more races. 
  * Otherwise, 
    1. If a≤ v, the probability of the item drawn becomes 0 and the item is no longer a valid item for all the further draws, reducing x by 1. Moreover, the reduced probability a is distributed equally among the other remaining valid items. 
    2. If a > v, the probability of the item drawn reduces by v and the reduced probability is distributed equally among the other valid items. 



For example,

  * If (c,m,p)=(0.2,0.1,0.7) and v=0.1, after drawing Cash, the new probabilities will be (0.1,0.15,0.75). 
  * If (c,m,p)=(0.1,0.2,0.7) and v=0.2, after drawing Cash, the new probabilities will be (Invalid,0.25,0.75). 
  * If (c,m,p)=(0.2,Invalid,0.8) and v=0.1, after drawing Cash, the new probabilities will be (0.1,Invalid,0.9). 
  * If (c,m,p)=(0.1,Invalid,0.9) and v=0.2, after drawing Cash, the new probabilities will be (Invalid,Invalid,1.0). 



You need the cars of Rivals. So, you need to find the expected number of races that you must play in order to draw a pink slip.

Input

The first line of input contains a single integer t (1≤ t≤ 10) — the number of test cases.

The first and the only line of each test case contains four real numbers c, m, p and v (0 < c,m,p < 1, c+m+p=1, 0.1≤ v≤ 0.9).

Additionally, it is guaranteed that each of c, m, p and v have at most 4 decimal places.

Output

For each test case, output a single line containing a single real number — the expected number of races that you must play in order to draw a Pink Slip.

Your answer is considered correct if its absolute or relative error does not exceed 10^{-6}.

Formally, let your answer be a, and the jury's answer be b. Your answer is accepted if and only if \frac{|a - b|}{max{(1, |b|)}} ≤ 10^{-6}.

Example

Input


4
0.2 0.2 0.6 0.2
0.4 0.2 0.4 0.8
0.4998 0.4998 0.0004 0.1666
0.3125 0.6561 0.0314 0.2048


Output


1.532000000000
1.860000000000
5.005050776521
4.260163673896

Note

For the first test case, the possible drawing sequences are: 

  * P with a probability of 0.6; 
  * CP with a probability of 0.2⋅ 0.7 = 0.14; 
  * CMP with a probability of 0.2⋅ 0.3⋅ 0.9 = 0.054; 
  * CMMP with a probability of 0.2⋅ 0.3⋅ 0.1⋅ 1 = 0.006; 
  * MP with a probability of 0.2⋅ 0.7 = 0.14; 
  * MCP with a probability of 0.2⋅ 0.3⋅ 0.9 = 0.054; 
  * MCCP with a probability of 0.2⋅ 0.3⋅ 0.1⋅ 1 = 0.006. 

So, the expected number of races is equal to 1⋅ 0.6 + 2⋅ 0.14 + 3⋅ 0.054 + 4⋅ 0.006 + 2⋅ 0.14 + 3⋅ 0.054 + 4⋅ 0.006 = 1.532.

For the second test case, the possible drawing sequences are: 

  * P with a probability of 0.4; 
  * CP with a probability of 0.4⋅ 0.6 = 0.24; 
  * CMP with a probability of 0.4⋅ 0.4⋅ 1 = 0.16; 
  * MP with a probability of 0.2⋅ 0.5 = 0.1; 
  * MCP with a probability of 0.2⋅ 0.5⋅ 1 = 0.1. 



So, the expected number of races is equal to 1⋅ 0.4 + 2⋅ 0.24 + 3⋅ 0.16 + 2⋅ 0.1 + 3⋅ 0.1 = 1.86.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
PREC = 0.00000005

def floatEq(a, b):
	return abs(a-b) < PREC

def f(c, m, p, v, n):
	r = (n+1)*p
	if c is not None and (c < v or floatEq(c, v)):
		if m is not None:
			r += c*f(None, m+c/2.0, p+c/2.0, v, n+1)
		else:
			r += c*f(None, None, p+c, v, n+1)
	elif c is not None and c > v:
		if m is not None:
			r += c*f(c-v, m+v/2.0, p+v/2.0, v, n+1)
		else:
			r += c*f(c-v, None, p+v, v, n+1)
	if m is not None and (m < v or floatEq(m, v)):
		if c is not None:
			r += m*f(c+m/2.0, None, p+m/2.0, v, n+1)
		else:
			r += m*f(None, None, p+m, v, n+1)
	elif m is not None and m > v:
		if c is not None:
			r += m*f(c+v/2.0, m-v, p+v/2.0, v, n+1)
		else:
			r += m*f(None, m-v, p+v, v, n+1)
	return r
I=lambda: map(float, raw_input().split())
t = input()
for _ in xrange(t):
	c, m, p, v = I()
	print f(c, m, p, v, 0)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cneedforpinkslipsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        super().__init__(**params)
    
    def case_generator(self):
        # 随机选择生成类型：普通案例、含极值p、初始值触发无效条件
        case_type = random.choice(['normal', 'extreme_p', 'low_prob'])
        
        if case_type == 'normal':
            # 标准生成方式
            a = random.randint(1, 9998)
            max_b = 10000 - a - 1  # 保证至少留1给d
            b = random.randint(1, max_b)
            d = 10000 - a - b
            c = round(a / 10000, 4)
            m = round(b / 10000, 4)
            p = round(d / 10000, 4)
            v = round(random.randint(1000, 9000) / 10000, 4)
            
        elif case_type == 'extreme_p':
            # 生成极小的p值（类似第三个测试案例）
            p = round(random.choice([0.0001, 0.0002, 0.0003, 0.0004]), 4)
            remaining = round(1 - p, 4)
            c = round(remaining * random.uniform(0.4, 0.6), 4)
            m = round(remaining - c, 4)
            v = round(random.uniform(0.15, 0.25), 4)
            
        elif case_type == 'low_prob':
            # 生成会立即触发无效条件的案例
            v_val = round(random.uniform(0.1, 0.3), 4)
            target = random.choice(['c', 'm'])
            if target == 'c':
                c = round(random.uniform(0.01, v_val), 4)
                remaining = 1 - c
                m = round(remaining * random.uniform(0.3, 0.7), 4)
                p = round(remaining - m, 4)
            else:
                m = round(random.uniform(0.01, v_val), 4)
                remaining = 1 - m
                c = round(remaining * random.uniform(0.3, 0.7), 4)
                p = round(remaining - c, 4)
            v = v_val

        # 校验并修正浮点精度
        total = round(c + m + p, 4)
        if total != 1.0:
            p = round(1 - c - m, 4)
        
        return {
            'c': c,
            'm': m,
            'p': p,
            'v': v
        }
    
    @staticmethod
    def prompt_func(question_case):
        c = question_case['c']
        m = question_case['m']
        p = question_case['p']
        v = question_case['v']
        prompt = f"""你刚刚击败了一个黑名单对手，获得了一个抽取奖励纸条的机会。初始时有三个有效纸条：现金纸条（Cash Slip，初始概率为{c:.4f}）、扣押解除标记（Impound Strike Release Marker，初始概率为{m:.4f}）和对手车辆的粉色纸条（Pink Slip，初始概率为{p:.4f}）。每次比赛获胜后，你可以抽取一张纸条。只要未抽到粉色纸条，你就可以继续参赛。每次抽取后，根据以下规则调整剩余纸条的概率：

1. 如果抽到的是粉色纸条，游戏结束，不再进行任何比赛。
2. 如果抽到的是其他纸条（如现金或解除标记）：
   - 若该纸条当前的抽取概率a小于等于波动因子v（当前v={v:.4f}），则该纸条变为无效（概率变为0），其原有概率平均分配给剩余的合法纸条。
   - 若该纸条的概率a大于v，则其概率减少v，减少的部分平均分配给其他合法纸条。

你的任务是计算需要进行的比赛的期望次数（即抽到粉色纸条之前的总抽取次数）。

输入数据为四个实数：{c:.4f} {m:.4f} {p:.4f} {v:.4f}。请严格根据上述规则计算期望值，并确保答案的绝对或相对误差不超过1e-6。答案必须包含足够多的小数位（例如，类似1.532000000000的格式）。

请将计算出的期望值放在[answer]和[/answer]的标签内。例如：[answer]1.532000000000[/answer]。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*([\d.]+)\s*\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        try:
            return float(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = cls._calculate_expected(identity['c'], identity['m'], identity['p'], identity['v'])
        if solution is None:
            return False
        try:
            solution_float = float(solution)
        except:
            return False
        # 双重验证机制
        abs_error = abs(solution_float - expected)
        if abs_error < 1e-6:
            return True
        rel_error = abs_error / max(1.0, abs(expected))
        return rel_error <= 1e-6
    
    @staticmethod
    def _calculate_expected(c_val, m_val, p_val, v_val):
        PREC = 1e-8

        def float_eq(a, b):
            return abs(a - b) < PREC

        memo = {}
        
        def f(c, m, p, n):
            key = (c if c is None else round(c,8), 
                   m if m is None else round(m,8), 
                   n)
            if key in memo:
                return memo[key]
            
            total = (n+1)*p
            remaining = []
            if c is not None:
                remaining.append(('c', c))
            if m is not None:
                remaining.append(('m', m))
            
            for item, value in remaining:
                if item == 'c':
                    current_c = value
                    if current_c < v_val or float_eq(current_c, v_val):
                        # Case 1: becomes invalid
                        if m is not None:
                            delta = current_c / 2
                            contribution = current_c * f(None, m+delta, p+delta, n+1)
                        else:
                            contribution = current_c * f(None, None, p+current_c, n+1)
                    else:
                        # Case 2: reduce by v
                        new_c = current_c - v_val
                        if m is not None:
                            delta = v_val / 2
                            contribution = current_c * f(new_c, m+delta, p+delta, n+1)
                        else:
                            contribution = current_c * f(new_c, None, p+v_val, n+1)
                    total += contribution
                elif item == 'm':
                    current_m = value
                    if current_m < v_val or float_eq(current_m, v_val):
                        # Case 1: becomes invalid
                        if c is not None:
                            delta = current_m / 2
                            contribution = current_m * f(c+delta, None, p+delta, n+1)
                        else:
                            contribution = current_m * f(None, None, p+current_m, n+1)
                    else:
                        # Case 2: reduce by v
                        new_m = current_m - v_val
                        if c is not None:
                            delta = v_val / 2
                            contribution = current_m * f(c+delta, new_m, p+delta, n+1)
                        else:
                            contribution = current_m * f(None, new_m, p+v_val, n+1)
                    total += contribution
            memo[key] = total
            return total
        
        c = c_val if c_val > 1e-8 else None
        m = m_val if m_val > 1e-8 else None
        p = p_val
        
        return f(c, m, p, 0)
