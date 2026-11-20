"""# 

### 谜题描述
Mr. Funt now lives in a country with a very specific tax laws. The total income of mr. Funt during this year is equal to n (n ≥ 2) burles and the amount of tax he has to pay is calculated as the maximum divisor of n (not equal to n, of course). For example, if n = 6 then Funt has to pay 3 burles, while for n = 25 he needs to pay 5 and if n = 2 he pays only 1 burle.

As mr. Funt is a very opportunistic person he wants to cheat a bit. In particular, he wants to split the initial n in several parts n1 + n2 + ... + nk = n (here k is arbitrary, even k = 1 is allowed) and pay the taxes for each part separately. He can't make some part equal to 1 because it will reveal him. So, the condition ni ≥ 2 should hold for all i from 1 to k.

Ostap Bender wonders, how many money Funt has to pay (i.e. minimal) if he chooses and optimal way to split n in parts.

Input

The first line of the input contains a single integer n (2 ≤ n ≤ 2·109) — the total year income of mr. Funt.

Output

Print one integer — minimum possible number of burles that mr. Funt has to pay as a tax.

Examples

Input

4


Output

2


Input

27


Output

3

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())

def prime(n):
	if n<=3:
		return True
	for i in range(2,int(n**(0.5))+1):
		if n%i==0:
			return False
	return True

if prime(n):
	print 1
elif n % 2 == 0:
	print 2
elif prime(n-2):
	print 2
else:
	print 3
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Btaxesbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10**9):
        """
        参数校验优化，支持更大范围的n值生成
        """
        if n_min < 2:
            raise ValueError("n_min must be ≥2")
        if n_max < n_min:
            raise ValueError("n_max must be ≥n_min")
        
        self.n_min = n_min
        self.n_max = n_max
    
    def case_generator(self):
        """主动构造四类典型案例，保证覆盖率"""
        def is_prime(m):
            if m <= 1:
                return False
            if m <=3:
                return True
            if m % 2 ==0 or m %3 ==0:
                return False
            i = 5
            w = 2
            while i*i <= m:
                if m%i ==0:
                    return False
                i += w
                w = 6 - w
            return True
        
        # 主动生成四类案例的平衡策略
        case_type = random.choice([
            'prime',         # 质数案例
            'even_composite',# 偶合数
            'odd_case2',     # 奇合数(n-2是质数)
            'odd_case3'      # 奇合数(n-2是合数)
        ])
        
        max_attempts = 1000
        for _ in range(max_attempts):
            # 动态调整生成策略
            if case_type == 'prime':
                # 生成随机质数
                n = random.randint(max(2, self.n_min), self.n_max)
                if is_prime(n):
                    return {'n': n, 'correct_answer': 1}
                
            elif case_type == 'even_composite':
                # 生成至少有两个质因子的偶数
                n = 2 * random.randint(2, self.n_max//2)
                if n >= 2 and not is_prime(n):
                    return {'n': n, 'correct_answer': 2}
            
            elif case_type == 'odd_case2':
                # 生成奇合数并满足n-2是质数
                base_prime = random.choice([3,5,7,11,13,17,19,23,29,31])
                n = base_prime + 2
                if n % 2 == 1 and not is_prime(n) and is_prime(base_prime):
                    return {'n': n, 'correct_answer': 2}
                # 动态生成
                candidate = random.randint(max(3, self.n_min), self.n_max)
                if candidate%2 ==1 and not is_prime(candidate) and is_prime(candidate-2):
                    return {'n': candidate, 'correct_answer': 2}
            
            elif case_type == 'odd_case3':
                # 确保生成正确结果为3的案例
                candidates = [27, 35, 45, 49, 55, 81, 875, 12345]
                for n in candidates:
                    if self.n_min <= n <= self.n_max:
                        if not is_prime(n) and n%2 ==1 and not is_prime(n-2):
                            return {'n': n, 'correct_answer': 3}
                # 动态生成
                candidate = random.randint(max(9, self.n_min), self.n_max)
                if candidate%2 ==1 and not is_prime(candidate) and not is_prime(candidate-2):
                    return {'n': candidate, 'correct_answer': 3}
        
        # Fallback机制：确保至少返回有效案例
        return {'n': 4, 'correct_answer': 2}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        return f"""根据俄罗斯税法规定，Funt先生的年收入为{n} burles，需要通过分割收入来最小化税款。规则如下：

1. 将总金额分割为k个整数（k≥1），每个部分≥2
2. 每个部分的税款为其最大真因子（即除自身外的最大约数）
3. 最终税款为各部分税款之和

请计算最小可能的税款金额，并将最终答案置于[answer]标签内，如：[answer]5[/answer]。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1].strip())
            except (ValueError, TypeError):
                return None
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer')
