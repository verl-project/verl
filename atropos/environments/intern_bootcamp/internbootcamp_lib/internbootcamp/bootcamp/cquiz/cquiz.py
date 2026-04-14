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
n,m,k=map(int,raw_input().split())
y=n%k
x=n/k
rem=10**9+9
if m<=x*(k-1)+y:
    print m%rem
else:
    z=x*(k-1)+y-m
    z=z*-1
    ans=0
    p=(z*k)%rem
    ans=(ans+((pow(2,z+1,rem)-2)*k)%rem)%rem
    ans=(ans+m-p)%rem
    print ans%rem
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

MOD = 10**9 + 9

class Cquizbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        参数调整支持大范围数值生成
        """
        self.max_n = params.get('max_n', 10**12)  # 允许生成题目上限的千倍规模
        self.max_k = params.get('max_k', 10**12)
    
    def case_generator(self):
        # 确保覆盖k=2的边界情况
        k = random.choice([2] + [random.randint(2, min(self.max_k, 10**5)) for _ in range(4)])
        n = random.randint(k, min(self.max_n, 10**12))
        m = random.randint(0, n)

        # 正确性计算保持不变
        y = n % k
        x = n // k
        if m <= x * (k-1) + y:
            correct_ans = m % MOD
        else:
            z = m - (x * (k-1) + y)
            part1 = (pow(2, z+1, MOD) - 2) * k % MOD
            part2 = (m - z * k) % MOD
            correct_ans = (part1 + part2) % MOD
        
        return {
            'n': n,
            'm': m,
            'k': k,
            'correct_ans': correct_ans
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        # 保持原问题描述逻辑不变
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        return f"""你是算法竞赛专家，需要解决以下数学问题：

**问题描述**：
Manao参加了一个包含{n}个问题的测试。每个正确答案得1分并增加连续正确计数器。当计数器达到{k}时，得分会先加1分再翻倍，然后计数器重置。错误答案会重置计数器。已知Manao正确回答了{m}题，求可能的最小分数模1000000009的结果。

**输入参数**：
- 总题数 n = {n}
- 正确回答数 m = {m}
- 连续要求 k = {k}

**要求**：
1. 计算所有可能回答顺序中的最小分数
2. 答案必须模1000000009
3. 将最终答案放在[answer]和[/answer]标签之间

示例格式：[answer]123[/answer]"""

    @staticmethod
    def extract_output(output):
        # 保持提取逻辑不变
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 保持验证逻辑不变
        return solution == identity['correct_ans']
