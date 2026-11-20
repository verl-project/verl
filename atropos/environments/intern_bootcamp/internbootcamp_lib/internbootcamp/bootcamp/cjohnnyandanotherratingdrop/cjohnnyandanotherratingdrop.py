"""# 

### 谜题描述
The last contest held on Johnny's favorite competitive programming platform has been received rather positively. However, Johnny's rating has dropped again! He thinks that the presented tasks are lovely, but don't show the truth about competitors' skills.

The boy is now looking at the ratings of consecutive participants written in a binary system. He thinks that the more such ratings differ, the more unfair is that such people are next to each other. He defines the difference between two numbers as the number of bit positions, where one number has zero, and another has one (we suppose that numbers are padded with leading zeros to the same length). For example, the difference of 5 = 101_2 and 14 = 1110_2 equals to 3, since 0101 and 1110 differ in 3 positions. Johnny defines the unfairness of the contest as the sum of such differences counted for neighboring participants.

Johnny has just sent you the rating sequence and wants you to find the unfairness of the competition. You have noticed that you've got a sequence of consecutive integers from 0 to n. That's strange, but the boy stubbornly says that everything is right. So help him and find the desired unfairness for received numbers.

Input

The input consists of multiple test cases. The first line contains one integer t (1 ≤ t ≤ 10 000) — the number of test cases. The following t lines contain a description of test cases.

The first and only line in each test case contains a single integer n (1 ≤ n ≤ 10^{18}).

Output

Output t lines. For each test case, you should output a single line with one integer — the unfairness of the contest if the rating sequence equals to 0, 1, ..., n - 1, n.

Example

Input


5
5
7
11
1
2000000000000


Output


8
11
19
1
3999999999987

Note

For n = 5 we calculate unfairness of the following sequence (numbers from 0 to 5 written in binary with extra leading zeroes, so they all have the same length): 

  * 000 
  * 001 
  * 010 
  * 011 
  * 100 
  * 101 



The differences are equal to 1, 2, 1, 3, 1 respectively, so unfairness is equal to 1 + 2 + 1 + 3 + 1 = 8.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
if sys.subversion[0] == \"PyPy\":
    import io, atexit
    sys.stdout = io.BytesIO()
    atexit.register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))
    
    sys.stdin = io.BytesIO(sys.stdin.read())
    input = lambda: sys.stdin.readline().rstrip()
 
RS = raw_input
RI = lambda x=int: map(x,RS().split())
RN = lambda x=int: x(RS())
''' ...................................................................... '''

for _ in xrange(RN()):
    n = RN()
    ans = 0
    while n:
        ans += n
        n>>=1
    print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cjohnnyandanotherratingdropbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10**18, seed=None):
        self.min_n = min_n
        self.max_n = max_n
        self.rng = random.Random(seed)
    
    def case_generator(self):
        # 生成策略优化：70%普通随机数，20%边界值，10%特殊模式
        if self.rng.random() < 0.7:
            n = self.rng.randint(self.min_n, self.max_n)
        elif self.rng.random() < 0.5:
            n = self.rng.choice([self.min_n, self.max_n])
        else:
            # 生成全1模式或2^k模式
            bits = self.rng.randint(1, 60)
            n = (1 << bits) - 1 if self.rng.random() < 0.5 else (1 << bits)
            n = min(max(n, self.min_n), self.max_n)
        return {"n": n}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["n"]
        return f"""## 编程竞赛不公平性计算问题

**任务背景**：
给定从0到{n}的连续整数序列（共{n+1}个数），计算所有相邻数对的二进制差异总和。

**规则说明**：
1. 二进制对齐：所有数字转换为相同位数的二进制表示（较短数补前导零）
2. 差异计算：每对相邻数比较每一位，统计不同位的总数
3. 相邻对数：共有{n}对相邻数（0与1, 1与2, ..., {n-1}与{n}）

**示例说明**：
当n=5时序列为：000, 001, 010, 011, 100, 101
差异计算：1+2+1+3+1=8

**当前输入**：
n = {n}

**答案要求**：
将最终结果用[answer]标签包裹，例如：[answer]8[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强版数字提取，处理各类异常情况
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_match = matches[-1].strip()
        numbers = re.findall(r'-?\d+', last_match.replace(',', ''))  # 处理千分位分隔符
        
        if not numbers:
            return None
        
        try:
            return int(numbers[-1])
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 数学验证逻辑加固
        n = identity["n"]
        if not isinstance(solution, int) or solution < 0:
            return False
        
        correct = 0
        current = n
        while current > 0:
            correct += current
            current >>= 1
        return solution == correct
