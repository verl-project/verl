"""# 

### 谜题描述
Right now she actually isn't. But she will be, if you don't solve this problem.

You are given integers n, k, A and B. There is a number x, which is initially equal to n. You are allowed to perform two types of operations: 

  1. Subtract 1 from x. This operation costs you A coins. 
  2. Divide x by k. Can be performed only if x is divisible by k. This operation costs you B coins. 

What is the minimum amount of coins you have to pay to make x equal to 1?

Input

The first line contains a single integer n (1 ≤ n ≤ 2·109).

The second line contains a single integer k (1 ≤ k ≤ 2·109).

The third line contains a single integer A (1 ≤ A ≤ 2·109).

The fourth line contains a single integer B (1 ≤ B ≤ 2·109).

Output

Output a single integer — the minimum amount of coins you have to pay to make x equal to 1.

Examples

Input

9
2
3
1


Output

6


Input

5
5
2
20


Output

8


Input

19
3
4
2


Output

12

Note

In the first testcase, the optimal strategy is as follows: 

  * Subtract 1 from x (9 → 8) paying 3 coins. 
  * Divide x by 2 (8 → 4) paying 1 coin. 
  * Divide x by 2 (4 → 2) paying 1 coin. 
  * Divide x by 2 (2 → 1) paying 1 coin. 



The total cost is 6 coins.

In the second test case the optimal strategy is to subtract 1 from x 4 times paying 8 coins in total.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, k, A, B = input(), input(), input(), input()

now = n
som = 0
while True:
    if now == 1:
        break

    kalan = now % k
    if k == 1:
        som += (now-1) * A
        break
    else:
        if kalan == 0:
            if (now - now/k) * A < B:
                som += (now - now/k) * A
                now = now / k
            else:
                som += B
                now = now / k
        else:
            if now < k:
                som += (now-1) * A
                break
            else:
                som += kalan * A
                now = now - kalan


print som
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def compute_min_coins(n, k, A, B):
    if n == 1:
        return 0
    total = 0
    current = n
    while current > 1:
        if k == 1:
            total += (current - 1) * A
            break
        remainder = current % k
        if remainder == 0:
            if (current - current // k) * A < B:
                total += (current - current // k) * A
                current = current // k
            else:
                total += B
                current = current // k
        else:
            if current < k:
                total += (current - 1) * A
                break
            else:
                total += remainder * A
                current -= remainder
    return total

class Bourtanyaiscryingoutloudbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.params = {
            'n_min': 1,
            'n_max': 2 * 10**9,
            'k_min': 1,  # 允许k=1的情况，增加案例多样性
            'k_max': 2 * 10**9,
            'A_min': 1,
            'A_max': 2 * 10**9,
            'B_min': 1,
            'B_max': 2 * 10**9
        }
        self.params.update(params)
    
    def case_generator(self):
        n = random.randint(self.params['n_min'], self.params['n_max'])
        k = random.randint(self.params['k_min'], self.params['k_max'])
        A = random.randint(self.params['A_min'], self.params['A_max'])
        B = random.randint(self.params['B_min'], self.params['B_max'])
        correct_answer = compute_min_coins(n, k, A, B)
        return {
            'n': n,
            'k': k,
            'A': A,
            'B': B,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        A = question_case['A']
        B = question_case['B']
        prompt = (
            f"You are given the integers n={n}, k={k}, A={A}, and B={B}. "
            "You need to find the minimum amount of coins required to reduce x, starting from x=n, to x=1. "
            "You can perform two operations:\n"
            "1. Subtract 1 from x, which costs A coins.\n"
            "2. Divide x by k, which costs B coins, but only if x is divisible by k.\n"
            "Please provide the minimum number of coins required. "
            "Put your answer within [answer] tags.\n"
            "For example, if the answer is 6, write: [answer]6[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        start_tag = "[answer]"
        end_tag = "[/answer]"
        parts = output.split(start_tag)
        if len(parts) < 2:
            return None
        last_part = parts[-1]
        end_index = last_part.find(end_tag)
        if end_index == -1:
            return None
        answer_str = last_part[:end_index].strip()
        try:
            return int(answer_str)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        return solution == identity['correct_answer']

# 示例使用
if __name__ == "__main__":
    bootcamp = Bourtanyaiscryingoutloudbootcamp()
    case = bootcamp.case_generator()
    prompt = Bourtanyaiscryingoutloudbootcamp.prompt_func(case)
    print("Prompt:", prompt)
    # 假设模型返回的回答
    response = "The minimum coins needed are [answer]6[/answer]."
    extracted = Bourtanyaiscryingoutloudbootcamp.extract_output(prompt + response)
    print("Extracted Answer:", extracted)
    score = Bourtanyaiscryingoutloudbootcamp.verify_score(extracted, case)
    print("Score:", score)
