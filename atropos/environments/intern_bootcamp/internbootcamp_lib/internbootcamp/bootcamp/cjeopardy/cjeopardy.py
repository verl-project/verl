"""# 

### 谜题描述
'Jeopardy!' is an intellectual game where players answer questions and earn points. Company Q conducts a simplified 'Jeopardy!' tournament among the best IT companies. By a lucky coincidence, the old rivals made it to the finals: company R1 and company R2. 

The finals will have n questions, m of them are auction questions and n - m of them are regular questions. Each question has a price. The price of the i-th question is ai points. During the game the players chose the questions. At that, if the question is an auction, then the player who chose it can change the price if the number of his current points is strictly larger than the price of the question. The new price of the question cannot be less than the original price and cannot be greater than the current number of points of the player who chose the question. The correct answer brings the player the points equal to the price of the question. The wrong answer to the question reduces the number of the player's points by the value of the question price.

The game will go as follows. First, the R2 company selects a question, then the questions are chosen by the one who answered the previous question correctly. If no one answered the question, then the person who chose last chooses again.

All R2 employees support their team. They want to calculate what maximum possible number of points the R2 team can get if luck is on their side during the whole game (they will always be the first to correctly answer questions). Perhaps you are not going to be surprised, but this problem was again entrusted for you to solve.

Input

The first line contains two space-separated integers n and m (1 ≤ n, m ≤ 100; m ≤ min(n, 30)) — the total number of questions and the number of auction questions, correspondingly. The second line contains n space-separated integers a1, a2, ..., an (1 ≤ ai ≤ 107) — the prices of the questions. The third line contains m distinct integers bi (1 ≤ bi ≤ n) — the numbers of auction questions. Assume that the questions are numbered from 1 to n.

Output

In the single line, print the answer to the problem — the maximum points the R2 company can get if it plays optimally well. It is guaranteed that the answer fits into the integer 64-bit signed type.

Examples

Input

4 1
1 3 7 5
3


Output

18


Input

3 2
10 3 8
2 3


Output

40


Input

2 2
100 200
1 2


Output

400

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map ( int, raw_input ( ).split ( ) )
a = map ( int, raw_input ( ).split ( ) )
b = map ( int, raw_input ( ).split ( ) )
visited = [ 0 for i in range(10000) ]

c = []
for i in range(m):
	visited[b[i]-1] = 1
	c.append ( a[b[i]-1] )

c.sort ( )
c.reverse ( )

sum = 0
for i in range(n):
	if not visited[i]:
		sum += a[i]

for i in range(m):
	if sum > c[i]:
		sum *= 2
	else:
		sum += c[i]

print sum
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cjeopardybootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=100, a_min=1, a_max=10**7, m_min=1, m_max=None):
        self.n_min = max(n_min, 1)
        self.n_max = n_max
        self.a_min = a_min
        self.a_max = a_max
        self.m_min = max(m_min, 1)
        self.m_max = m_max
    
    def case_generator(self):
        # Generate valid n and ensure constraints
        n = random.randint(self.n_min, self.n_max)
        
        # Calculate valid m range
        possible_m_max = min(n, 30)
        m_max = min(self.m_max, possible_m_max) if self.m_max is not None else possible_m_max
        m_min = max(self.m_min, 1)
        
        # Clamp m_min/m_max to valid range
        m_min = min(m_min, possible_m_max)
        m_max = min(m_max, possible_m_max)
        if m_min > m_max:  # Handle invalid user parameters
            m_min = 1
            m_max = possible_m_max
        
        m = random.randint(m_min, m_max) if m_min <= m_max else 1
        
        # Generate questions and auction markers
        a = [random.randint(self.a_min, self.a_max) for _ in range(n)]
        b = random.sample(range(1, n+1), m)  # 1-based question numbers
        
        # Calculate expected maximum sum
        auction_prices = [a[i-1] for i in b]
        non_auction_sum = sum(price for idx, price in enumerate(a, 1) if idx not in b)
        
        sorted_auction = sorted(auction_prices, reverse=True)
        current_sum = non_auction_sum
        for price in sorted_auction:
            current_sum = current_sum * 2 if current_sum > price else current_sum + price
        
        return {
            'n': n,
            'm': m,
            'a': a,
            'b': sorted(b),  # 输出排序后的拍卖题号方便阅读
            'expected_sum': current_sum
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return (
            "Company R2 and R1 are competing in a Jeopardy-style tournament final. Parameters:\n"
            f"- Total questions: {question_case['n']}\n"
            f"- Auction questions: {question_case['m']}\n"
            f"- Prices (questions 1-{question_case['n']}): {', '.join(map(str, question_case['a']))}\n"
            f"- Auction question numbers: {', '.join(map(str, question_case['b']))}\n\n"
            "Game Rules:\n"
            "1. Auction Questions: When R2 selects these, if their current points > original price,\n"
            "   they can set the price to any value between original and current points. Correct answer\n"
            "   grants the new price, wrong answer deducts the original price.\n"
            "2. Regular Questions: Directly gain/lose the original price.\n\n"
            "Assuming R2 always answers first and correctly with optimal strategy, find the maximum\n"
            "possible points. Put your answer in [answer] tags like: [answer]12345[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['expected_sum']
        except (ValueError, TypeError, KeyError):
            return False
