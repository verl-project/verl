"""# 

### 谜题描述
Prof. Vasechkin wants to represent positive integer n as a sum of addends, where each addends is an integer number containing only 1s. For example, he can represent 121 as 121=111+11+–1. Help him to find the least number of digits 1 in such sum.

Input

The first line of the input contains integer n (1 ≤ n < 1015).

Output

Print expected minimal number of digits 1.

Examples

Input

121


Output

6

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def solve(n,p):
	aparitii = n/a[p];
	if(n%a[p]==0):
		return aparitii*p;
	sol1 = aparitii * p + solve(n%a[p],p-1);
	sol2 = (aparitii+1) * p + solve(a[p]*(aparitii+1)-n,p-1);
	return min(sol1,sol2);
		

a = [];
n = int(input());

a.append(0);
for i in range(1,18):
	a.append(a[i-1]*10+1);

print solve(n,16)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from functools import lru_cache
from bootcamp import Basebootcamp

class Conebasedarithmeticbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10**15-1):
        self.min_n = min_n
        self.max_n = max_n
        # 预处理生成1~17位的全1数
        self.a = [0]
        for i in range(1, 17+1):
            self.a.append(self.a[-1] * 10 + 1)
    
    def case_generator(self):
        # 生成三类测试用例：全1数、接近全1数的边缘值、随机数
        rand = random.random()
        if rand < 0.2:  # 全1数
            digits = random.randint(1, 17)
            n = self.a[digits]
        elif rand < 0.4:  # 全1数±1
            digits = random.randint(2, 16)
            base = self.a[digits]
            n = base + random.choice([-1, 1])
        else:  # 普通随机数
            n = random.randint(self.min_n, min(self.max_n, 10**17))
        return {'n': n}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return (
            f"将整数 {n} 表示为全1数字的加减组合，找到最少需要使用的1的总个数。\n"
            "规则说明：\n"
            "1. 每个加数/减数必须是由连续1组成的整数（如11, 111等）\n"
            "2. 允许使用加减法组合，例如：121 = 111 + 11 - 1\n"
            "3. 要求最终解中使用的1字符总数最少\n\n"
            "请给出最少需要的1的总个数，并置于[answer]标签内，例如：[answer]6[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        try:
            return int(matches[-1].strip()) if matches else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            instance = cls()  # 创建带预计算数据的实例
            return int(solution) == instance.calculate_min_ones(n)
        except:
            return False

    @lru_cache(maxsize=None)
    def calculate_min_ones(self, n):
        def dfs(n, p):
            if p == 0:
                return float('inf') if n !=0 else 0
            
            current = self.a[p]
            if current == 0:
                return dfs(n, p-1)
            
            quotient, remainder = divmod(n, current)
            
            if remainder == 0:
                return quotient * p
            
            # 正向处理方案
            option1 = quotient * p + dfs(remainder, p-1)
            # 溢出处理方案（多用一个current）
            option2 = (quotient + 1) * p + dfs(current*(quotient+1)-n, p-1)
            
            return min(option1, option2)
        
        return dfs(n, p=len(self.a)-1)  # 从最大位数开始处理
