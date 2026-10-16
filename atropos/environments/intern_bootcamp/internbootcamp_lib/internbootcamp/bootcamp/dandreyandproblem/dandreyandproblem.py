"""# 

### 谜题描述
Andrey needs one more problem to conduct a programming contest. He has n friends who are always willing to help. He can ask some of them to come up with a contest problem. Andrey knows one value for each of his fiends — the probability that this friend will come up with a problem if Andrey asks him.

Help Andrey choose people to ask. As he needs only one problem, Andrey is going to be really upset if no one comes up with a problem or if he gets more than one problem from his friends. You need to choose such a set of people that maximizes the chances of Andrey not getting upset.

Input

The first line contains a single integer n (1 ≤ n ≤ 100) — the number of Andrey's friends. The second line contains n real numbers pi (0.0 ≤ pi ≤ 1.0) — the probability that the i-th friend can come up with a problem. The probabilities are given with at most 6 digits after decimal point.

Output

Print a single real number — the probability that Andrey won't get upset at the optimal choice of friends. The answer will be considered valid if it differs from the correct one by at most 10 - 9.

Examples

Input

4
0.1 0.2 0.3 0.8


Output

0.800000000000


Input

2
0.1 0.2


Output

0.260000000000

Note

In the first sample the best strategy for Andrey is to ask only one of his friends, the most reliable one.

In the second sample the best strategy for Andrey is to ask all of his friends to come up with a problem. Then the probability that he will get exactly one problem is 0.1·0.8 + 0.9·0.2 = 0.26.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

n=int(sys.stdin.readline().strip('\n'))
line=sys.stdin.readline().strip('\n').split(' ')
probs=map(lambda x:float(x),line)
probInvs=1
prob=0
while len(probs)>0 and probInvs>prob:
    #print prob,probInvs
    prob=(1-max(probs))*prob+max(probs)*probInvs
    probInvs=(1-max(probs))*probInvs
    probs.remove(max(probs))

print prob
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re

class Dandreyandproblembootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100, precision=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.min_n = min_n
        self.max_n = max_n
        self.precision = precision

    @staticmethod
    def compute_max_prob(probs):
        sorted_probs = sorted(probs, reverse=True)
        max_prob = 0.0
        product_inverse = 1.0
        current_prob = 0.0
        
        for p in sorted_probs:
            candidate = (1 - p)*current_prob + p*product_inverse
            if candidate > max_prob + 1e-15:  # 防止浮点误差误判
                max_prob = candidate
                product_inverse *= (1 - p)
                current_prob = max_prob
            else:
                break
        return max_prob

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        probs = [
            float(f"{random.uniform(0.0, 1.0):.6f}")  # 确保精确的6位小数
            for _ in range(n)
        ]
        return {
            'n': n,
            'probs': probs,
            'correct_answer': self.compute_max_prob(probs)
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        probs = question_case['probs']
        return (
            "Andrey needs to select friends to maximize the probability of getting exactly one problem.\n\n"
            "Problem Rules:\n"
            "1. Choose a subset of friends to ask\n"
            "2. The probability of success is calculated for exactly one friend succeeding\n"
            "3. Output must have at least 9 decimal places\n\n"
            f"Input:\n{question_case['n']}\n{' '.join(f'{p:.6f}' for p in probs)}\n\n"
            "Output format:\n[answer]probability[/answer]\n"
            "Example: [answer]0.260000000000[/answer]"
        )

    @staticmethod
    def extract_output(output):
        # 增强科学计数法支持
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
            
        try:
            # 处理千分位分隔符和科学计数法
            last_match = matches[-1].strip().replace(',', '')
            if 'e' in last_match or 'E' in last_match:
                return float(f"{float(last_match):.12f}")
            return float(last_match)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = identity['correct_answer']
            return abs(solution - expected) <= 1e-9 + 1e-12  # 增强容错性
        except:
            return False
