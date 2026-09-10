"""# 

### 谜题描述
Alexander is learning how to convert numbers from the decimal system to any other, however, he doesn't know English letters, so he writes any number only as a decimal number, it means that instead of the letter A he will write the number 10. Thus, by converting the number 475 from decimal to hexadecimal system, he gets 11311 (475 = 1·162 + 13·161 + 11·160). Alexander lived calmly until he tried to convert the number back to the decimal number system.

Alexander remembers that he worked with little numbers so he asks to find the minimum decimal number so that by converting it to the system with the base n he will get the number k.

Input

The first line contains the integer n (2 ≤ n ≤ 109). The second line contains the integer k (0 ≤ k < 1060), it is guaranteed that the number k contains no more than 60 symbols. All digits in the second line are strictly less than n.

Alexander guarantees that the answer exists and does not exceed 1018.

The number k doesn't contain leading zeros.

Output

Print the number x (0 ≤ x ≤ 1018) — the answer to the problem.

Examples

Input

13
12


Output

12

Input

16
11311


Output

475

Input

20
999


Output

3789

Input

17
2016


Output

594

Note

In the first example 12 could be obtained by converting two numbers to the system with base 13: 12 = 12·130 or 15 = 1·131 + 2·130.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(input())
k = str(input())

INF = 10**19
pown = []
tr = len(str(n))
dp = [range(0,70) for i in range(0,70)]
for i in range(0,70) :
	for j in range(0,70) :
		dp[i][j] = (-1,-1)

pown.append(1)
for i in range(1,70) : 
	if pown[i-1] == INF :
		pown.append(INF)
		continue
	x = pown[i-1]
	kk = x*n
	if kk > INF :
		kk = INF
	pown.append(kk)

def check(f) :
	if len(f) > tr :
		return INF
	if int(f) >= n :
		return INF

	return int(f)

def DP(x,y) :
	if dp[x][y] != (-1,-1) :
		return dp[x][y]
	if x > y :
		dp[x][y] = (0,0)
		return dp[x][y]
	ret = (INF,0)
	for i in range(x,y) :
		a = DP(x,i)
		b = DP(i+1,y)
		pp = pown[b[1]]
		if pp == INF :
			continue
		p = b[0]+(a[0]* pp)

		if p < ret[0] :
			ret = (p,a[1]+b[1])

	f =\"\"
	for i in range(x,y+1) :
		f += k[i]

	e = check(f)
	#print x,y,ret,e
	dp[x][y] = ret
	if e < dp[x][y][0] :
		dp[x][y] = (e,1)

	return dp[x][y]


print DP(0,len(k)-1)[0]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dabilitytoconvertbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_n = params.get('min_n', 2)
        self.max_n = params.get('max_n', 100)  # 测试时采用较小范围
        self.max_k_length = params.get('max_k_length', 10)
        self.min_k_length = params.get('min_k_length', 1)

    def case_generator(self):
        for _ in range(1000):
            n = random.randint(self.min_n, self.max_n)
            allowed_digits = list(map(str, range(min(n, 10))))
            if not allowed_digits:
                allowed_digits = ['0']
            
            # 生成有效k_str（每个字符严格<n）
            k_length = random.randint(self.min_k_length, self.max_k_length)
            k_chars = [random.choice(allowed_digits) for _ in range(k_length)]
            k_str = ''.join(k_chars).lstrip('0') or '0'
            
            if len(k_str) > 60:
                continue
            
            # 计算正确答案
            try:
                expected_x = self.calculate_min_x(n, k_str)
                if expected_x is not None and 0 <= expected_x <= 1e18:
                    return {'n': n, 'k': k_str, 'expected_x': expected_x}
            except Exception as e:
                continue
        
        # 保底用例
        return {'n': 10, 'k': '0', 'expected_x': 0}

    @staticmethod
    def prompt_func(case):
        return f"""Alexander将十进制数转换为n进制时，用十进制数字代替字母。例如，当n=16时，475转换为'11311'。现在给定n={case['n']}和k={case['k']}，请找出最小的十进制数x，使得x转换为n进制的Alexander表示正好是k。答案请写在[answer]和[/answer]之间。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_x']

    def calculate_min_x(self, n, k_str):
        INF = 10**19
        length = len(k_str)
        if length == 0:
            return 0
        
        # 预处理n的位数阈值
        tr = len(str(n))
        
        # 初始化权值数组
        pown = [1] * 70
        for i in range(1, 70):
            pown[i] = pown[i-1] * n if pown[i-1] <= INF // n else INF
        
        # DP表：dp[i][j] = (min_value, digits_count)
        dp = [[(INF, 0) for _ in range(length)] for __ in range(length)]
        
        # 填充DP表
        for l in range(1, length+1):
            for i in range(length - l + 1):
                j = i + l - 1
                current_str = k_str[i:j+1]
                
                # 候选1：整个子串作为单个数字
                if len(current_str) <= tr:
                    num = int(current_str)
                    if num < n and num < dp[i][j][0]:
                        dp[i][j] = (num, 1)
                
                # 候选2：分割子串
                for mid in range(i, j):
                    left_val, left_len = dp[i][mid]
                    right_val, right_len = dp[mid+1][j]
                    if right_len >= len(pown) or pown[right_len] == INF:
                        continue
                    combined = left_val * pown[right_len] + right_val
                    if combined < dp[i][j][0] and combined <= INF:
                        dp[i][j] = (combined, left_len + right_len)
        
        return dp[0][length-1][0] if dp[0][length-1][0] <= 1e18 else None
