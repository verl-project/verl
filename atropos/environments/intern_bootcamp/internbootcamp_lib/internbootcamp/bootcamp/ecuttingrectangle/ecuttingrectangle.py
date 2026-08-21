"""# 

### 谜题描述
A rectangle with sides A and B is cut into rectangles with cuts parallel to its sides. For example, if p horizontal and q vertical cuts were made, (p + 1) ⋅ (q + 1) rectangles were left after the cutting. After the cutting, rectangles were of n different types. Two rectangles are different if at least one side of one rectangle isn't equal to the corresponding side of the other. Note that the rectangle can't be rotated, this means that rectangles a × b and b × a are considered different if a ≠ b.

For each type of rectangles, lengths of the sides of rectangles are given along with the amount of the rectangles of this type that were left after cutting the initial rectangle.

Calculate the amount of pairs (A; B) such as the given rectangles could be created by cutting the rectangle with sides of lengths A and B. Note that pairs (A; B) and (B; A) are considered different when A ≠ B.

Input

The first line consists of a single integer n (1 ≤ n ≤ 2 ⋅ 10^{5}) — amount of different types of rectangles left after cutting the initial rectangle.

The next n lines each consist of three integers w_{i}, h_{i}, c_{i} (1 ≤ w_{i}, h_{i}, c_{i} ≤ 10^{12}) — the lengths of the sides of the rectangles of this type and the amount of the rectangles of this type.

It is guaranteed that the rectangles of the different types are different.

Output

Output one integer — the answer to the problem.

Examples

Input

1
1 1 9


Output

3


Input

2
2 3 20
2 4 40


Output

6


Input

2
1 2 5
2 3 5


Output

0

Note

In the first sample there are three suitable pairs: (1; 9), (3; 3) and (9; 1).

In the second sample case there are 6 suitable pairs: (2; 220), (4; 110), (8; 55), (10; 44), (20; 22) and (40; 11).

Here the sample of cut for (20; 22).

<image>

The third sample has no suitable pairs.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()

w=[]
h=[]
c=[]
cntw={}
cnth={}
gcdC=0
cntC=0

def insert1(a,b,c):
	if not a in b :
		b[a]=c
	else :
		b[a]=b[a]+c

def gcd(a,b): 
	if a % b == 0 :
		return b
	else :
		return gcd(b,a%b)

for i in range(0, n):
	a,b,d = map(int, raw_input().split())
	w.append(a)
	h.append(b)
	c.append(d)
	insert1(a,cntw,d)
	insert1(b,cnth,d)
	cntC += d
	if gcdC == 0 :
		gcdC = d
	else :
		gcdC = gcd(gcdC, d)

for i in range(0, n):
	if cntw[w[i]] * cnth[h[i]] != cntC * c[i]:
		print 0
		exit()

ans = 0
i = 1
while (i * i <= gcdC) :
	if gcdC % i == 0 :
		ans += 1
		if i * i != gcdC :
			ans += 1
	i += 1

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from collections import defaultdict
from math import gcd
from bootcamp import Basebootcamp

class Ecuttingrectanglebootcamp(Basebootcamp):
    def __init__(self, **params):
        pass
    
    def case_generator(self):
        possible_d = [6, 12, 9, 4, 7, 8, 10]
        d = random.choice(possible_d)
        n = random.randint(1, 3)
        
        # 修正m_list生成逻辑，保证n个元素
        m_list = []
        current_gcd = 0
        for _ in range(n):
            if current_gcd == 0:
                m = random.randint(1, 5)
                current_gcd = m
            else:
                # 强制生成与current_gcd互质的m
                possible_m = [x for x in range(1, 6) if math.gcd(current_gcd, x) == 1]
                if not possible_m:  # 若无候选值，使用1（必互质）
                    m = 1
                else:
                    m = random.choice(possible_m)
                current_gcd = math.gcd(current_gcd, m)  # 更新gcd
            m_list.append(m)
        
        # 生成w_values和h_values确保长度n
        use_same_w = random.choice([True, False])
        w_values = []
        if use_same_w:
            w = random.randint(1, 1000)
            w_values = [w] * n
        else:
            used_w = set()
            while len(w_values) < n:
                w = random.randint(1, 1000)
                if w not in used_w:
                    used_w.add(w)
                    w_values.append(w)
        
        h_values = []
        used_h = set()
        while len(h_values) < n:
            h = random.randint(1, 1000)
            if h not in used_h:
                used_h.add(h)
                h_values.append(h)
        
        # 构造矩形列表
        rectangles = [{
            'w': w_values[i],
            'h': h_values[i],
            'c': d * m_list[i]
        } for i in range(n)]
        
        case = {'n': n, 'rectangles': rectangles}
        
        # 生成无效案例时不破坏列表结构
        if random.random() < 0.5 and n > 0:
            idx = random.randint(0, n-1)
            case['rectangles'][idx]['c'] += 1
        
        return case
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['n'])]
        for rect in question_case['rectangles']:
            input_lines.append(f"{rect['w']} {rect['h']} {rect['c']}")
        problem_input = "\n".join(input_lines)
        prompt = f"""你是一个数学问题解决专家，请解决以下问题：

问题描述：
给定一个初始矩形被切割后的所有小矩形信息，计算可能的原始矩形尺寸对(A, B)的数量。注意(A, B)和(B, A)视为不同的对（当A≠B时）。

输入格式：
第一行是一个整数n，表示小矩形的不同种类数。接下来的n行每行包含三个整数w_i, h_i, c_i，分别表示第i种小矩形的宽、高和数量。

输出格式：
输出一个整数，表示符合条件的(A, B)对的数量。

示例：
输入：
1
1 1 9
输出：
3

你的任务：
输入：
{problem_input}
请将答案放在[answer]和[/answer]标签之间。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        rectangles = identity['rectangles']
        if n != len(rectangles):
            return solution == 0  # 异常情况处理
        
        c_list = [r['c'] for r in rectangles]
        total_c = sum(c_list)
        
        cntw = defaultdict(int)
        cnth = defaultdict(int)
        for r in rectangles:
            cntw[r['w']] += r['c']
            cnth[r['h']] += r['c']
        
        valid = all(cntw[r['w']] * cnth[r['h']] == total_c * r['c'] for r in rectangles)
        if not valid:
            return solution == 0
        
        current_gcd = c_list[0]
        for c in c_list[1:]:
            current_gcd = math.gcd(current_gcd, c)
        
        def count_divisors(x):
            if x == 0:
                return 0
            cnt = 0
            sqrt_x = int(math.isqrt(x))
            for i in range(1, sqrt_x + 1):
                if x % i == 0:
                    cnt += 1 if i == x // i else 2
            return cnt
        
        correct = count_divisors(current_gcd)
        return solution == correct
