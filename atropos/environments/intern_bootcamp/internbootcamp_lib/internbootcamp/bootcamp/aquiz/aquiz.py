"""# 

### 谜题描述
Manao is taking part in a quiz. The quiz consists of n consecutive questions. A correct answer gives one point to the player. The game also has a counter of consecutive correct answers. When the player answers a question correctly, the number on this counter increases by 1. If the player answers a question incorrectly, the counter is reset, that is, the number on it reduces to 0. If after an answer the counter reaches the number k, then it is reset, and the player's score is doubled. Note that in this case, first 1 point is added to the player's score, and then the total score is doubled. At the beginning of the game, both the player's score and the counter of consecutive correct answers are set to zero.

Manao remembers that he has answered exactly m questions correctly. But he does not remember the order in which the questions came. He's trying to figure out what his minimum score may be. Help him and compute the remainder of the corresponding number after division by 1000000009 (109 + 9).

Input

The single line contains three space-separated integers n, m and k (2 ≤ k ≤ n ≤ 109; 0 ≤ m ≤ n).

Output

Print a single integer — the remainder from division of Manao's minimum possible score in the quiz by 1000000009 (109 + 9).

Examples

Input

5 3 2


Output

3


Input

5 4 2


Output

6

Note

Sample 1. Manao answered 3 questions out of 5, and his score would double for each two consecutive correct answers. If Manao had answered the first, third and fifth questions, he would have scored as much as 3 points.

Sample 2. Now Manao answered 4 questions. The minimum possible score is obtained when the only wrong answer is to the question 4.

Also note that you are asked to minimize the score and not the remainder of the score modulo 1000000009. For example, if Manao could obtain either 2000000000 or 2000000020 points, the answer is 2000000000 mod 1000000009, even though 2000000020 mod 1000000009 is a smaller number.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m,k = map(int, raw_input().split())
mod = 10**9+9
x = max(0, n/k-n+m)
p = pow(2,x+1,mod)-2
print (p*k+m-x*k)%mod
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Aquizbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params.copy()
        # 设置参数默认值
        self.params.setdefault('min_k', 2)
        self.params.setdefault('max_k', 10**9)
        self.params.setdefault('min_n', 2)
        self.params.setdefault('max_n', 10**9)
        super().__init__(**params)
    
    def case_generator(self):
        # 优先使用直接指定的参数
        if all(k in self.params for k in ('n', 'm', 'k')):
            n = self.params['n']
            m = self.params['m']
            k = self.params['k']
            # 参数验证
            if not (2 <= k <= n and 0 <= m <= n):
                raise ValueError("Invalid puzzle parameters")
            return {'n': n, 'm': m, 'k': k}
        
        # 随机生成符合规范的参数
        k = random.randint(
            self.params['min_k'],
            min(self.params['max_k'], self.params['max_n'])
        )
        n = random.randint(
            max(k, self.params['min_n']),
            self.params['max_n']
        )
        m = random.randint(0, n)
        return {'n': n, 'm': m, 'k': k}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        prompt = (
            f"## 谜题背景\nManao参加了一个有{n}道题的竞赛，其中答对{m}题。"
            f"\n\n## 计分规则\n- 每答对1题得1分\n- 连续答对计数达{k}次时："
            f"\n  1. 先加1分\n  2. 总分立即翻倍\n  3. 重置连续计数"
            f"\n\n## 你的任务\n找到Manao可能获得的最小分数，结果模1000000009。"
            f"\n\n## 答案格式\n将最终答案放在[answer]和[/answer]之间，例如：[answer]42[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 增强数字格式匹配（支持千分位逗号）
        matches = re.findall(r'\[answer\s*\]\s*(-?\d[\d,]*)\s*\[/answer\s*\]', output, re.IGNORECASE)
        if not matches:
            return None
        try:
            # 处理最后一个匹配项并转换格式
            return int(matches[-1].replace(',', '').strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        mod = 10**9 + 9
        n = identity['n']
        m = identity['m']
        k = identity['k']
        
        if m == 0:
            return solution == 0
        
        # 计算最大非加倍得分区域
        error_blocks = n - m + 1  # 错误答案分割出的区间数
        max_safe = error_blocks * (k - 1)
        
        if m <= max_safe:
            expected = m % mod
        else:
            # 需要处理加倍的得分
            excess = m - max_safe
            doubles = excess // k
            remainder = excess % k
            
            # 计算加倍得分部分
            pow_val = pow(2, doubles + 1, mod) - 2
            doubled_part = (pow_val * k) % mod
            
            # 计算剩余得分部分
            remainder_part = (max_safe + remainder) % mod
            
            expected = (doubled_part + remainder_part) % mod
        
        return solution % mod == expected
