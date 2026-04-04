"""# 

### 谜题描述
Consider a sequence [a1, a2, ... , an]. Define its prefix product sequence <image>.

Now given n, find a permutation of [1, 2, ..., n], such that its prefix product sequence is a permutation of [0, 1, ..., n - 1].

Input

The only input line contains an integer n (1 ≤ n ≤ 105).

Output

In the first output line, print \"YES\" if such sequence exists, or print \"NO\" if no such sequence exists.

If any solution exists, you should output n more lines. i-th line contains only an integer ai. The elements of the sequence should be different positive integers no larger than n.

If there are multiple solutions, you are allowed to print any of them.

Examples

Input

7


Output

YES
1
4
3
6
5
2
7


Input

6


Output

NO

Note

For the second sample, there are no valid sequences.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/python
# coding: utf-8

def isPrime(n):
  if n == 2:
    return True
  if n % 2 == 0:
    return False
  for i in xrange(3, (int)(n ** 0.5) + 1, 2):
    if n % i == 0:
      return False
  return True
n = input()
if n == 1:
  print 'YES\n1'
elif n == 4:
  print 'YES\n1\n3\n2\n4'
elif not isPrime(n):
  print 'NO'
else:
  print 'YES\n1'
  inv = [0, 1]
  for i in xrange(2, n):
    inv.append((n - n / i) * inv[n % i] % n)
    print i * inv[i - 1] % n
  print n
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Eprefixproductsequencebootcamp(Basebootcamp):
    def __init__(self, max_n=100000, min_n=1):
        """
        初始化训练场参数，确保n的取值范围合法。
        
        参数:
            max_n (int): 生成的n的最大值（默认1e5，须小于等于1e5）
            min_n (int): 生成的n的最小值（默认1，须大于等于1）
        """
        if min_n < 1:
            raise ValueError("min_n must be at least 1")
        if max_n > 10**5:
            raise ValueError("max_n cannot exceed 100000")
        if min_n > max_n:
            raise ValueError("min_n must be <= max_n")
        self.max_n = max_n
        self.min_n = min_n
    
    @staticmethod
    def is_prime(n):
        """优化质数判断逻辑，提升大数效率"""
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def case_generator(self):
        """
        生成n时确保覆盖所有边界条件：
        - 强制包含n=1,4以确保特殊案例覆盖率
        - 50%概率生成质数用于覆盖YES案例
        """
        candidates = []
        if self.min_n <= 1 <= self.max_n:
            candidates.append(1)
        if self.min_n <= 4 <= self.max_n:
            candidates.append(4)
        # 随机生成至少一个质数案例（若范围内存在）
        prime_candidate = self._find_prime_in_range()
        if prime_candidate:
            candidates.append(prime_candidate)
        
        # 随机选择n（优先特殊案例）
        if candidates and random.random() < 0.5:
            n = random.choice(candidates)
        else:
            n = random.randint(self.min_n, self.max_n)
        
        possible = n in (1, 4) or self.is_prime(n)
        return {'n': n, 'possible': possible}
    
    def _find_prime_in_range(self):
        """在允许范围内查找一个质数"""
        attempts = 0
        while attempts < 1000:
            n = random.randint(self.min_n, self.max_n)
            if self.is_prime(n):
                return n
            attempts += 1
        return None  # 未找到时不强制要求
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """优化问题描述的数学表达式清晰度"""
        n = question_case['n']
        problem_desc = (
            f"给定整数 n = {n}，寻找一个排列 P = [a₁, a₂, ..., aₙ] 满足：\n\n"
            "条件定义：\n"
            f"- 前缀积序列 B = [b₁, b₂, ..., bₙ]，其中 b_i = (a₁×a₂×...×a_i) mod {n}\n"
            f"- 要求 B 是 [0, 1, 2, ..., {n-1}] 的一个排列\n\n"
            "输出要求：\n"
            f"1. 第一行输出是否存在解（YES/NO）\n"
            f"2. 若存在解，输出{n}行具体排列（每行一个整数）\n\n"
            "答案格式要求：\n"
            "将完整输出包含在[answer]和[/answer]标记之间，例如：\n"
            "[answer]\n"
            "YES\n1\n3\n2\n4\n[/answer]"
        )
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        """增强正则表达式鲁棒性，允许换行和空格"""
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        # 清理首尾空白行
        return '\n'.join(line.strip() for line in last_match.splitlines() if line.strip())
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """添加模运算结果的严格验证"""
        n = identity['n']
        possible = identity['possible']
        lines = solution.strip().split('\n')
        if not lines:
            return False
        
        # 验证YES/NO与possible的一致性
        first_line = lines[0].strip().upper()
        if first_line == 'YES' and not possible:
            return False
        if first_line == 'NO' and possible:
            return False
        
        # 处理NO情况
        if first_line == 'NO':
            return len(lines) == 1
        
        # 处理YES情况
        if len(lines) != n + 1:
            return False  # 行数不匹配
        
        try:
            sequence = list(map(int, lines[1:n+1]))
        except ValueError:
            return False
        
        # 验证元素为1~n的排列
        if sorted(sequence) != list(range(1, n+1)):
            return False
        
        # 验证前缀积模运算结果
        prefix_mod = []
        current = 1
        for num in sequence:
            current = (current * num) % n
            prefix_mod.append(current)
        return sorted(prefix_mod) == list(range(n))
