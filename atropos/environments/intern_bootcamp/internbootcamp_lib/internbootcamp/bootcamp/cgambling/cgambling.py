"""# 

### 谜题描述
Two players A and B have a list of n integers each. They both want to maximize the subtraction between their score and their opponent's score. 

In one turn, a player can either add to his score any element from his list (assuming his list is not empty), the element is removed from the list afterward. Or remove an element from his opponent's list (assuming his opponent's list is not empty).

Note, that in case there are equal elements in the list only one of them will be affected in the operations above. For example, if there are elements \{1, 2, 2, 3\} in a list and you decided to choose 2 for the next turn, only a single instance of 2 will be deleted (and added to the score, if necessary). 

The player A starts the game and the game stops when both lists are empty. Find the difference between A's score and B's score at the end of the game, if both of the players are playing optimally.

Optimal play between two players means that both players choose the best possible strategy to achieve the best possible outcome for themselves. In this problem, it means that each player, each time makes a move, which maximizes the final difference between his score and his opponent's score, knowing that the opponent is doing the same.

Input

The first line of input contains an integer n (1 ≤ n ≤ 100 000) — the sizes of the list.

The second line contains n integers a_i (1 ≤ a_i ≤ 10^6), describing the list of the player A, who starts the game.

The third line contains n integers b_i (1 ≤ b_i ≤ 10^6), describing the list of the player B.

Output

Output the difference between A's score and B's score (A-B) if both of them are playing optimally.

Examples

Input

2
1 4
5 1


Output

0

Input

3
100 100 100
100 100 100


Output

0

Input

2
2 1
5 6


Output

-3

Note

In the first example, the game could have gone as follows: 

  * A removes 5 from B's list. 
  * B removes 4 from A's list. 
  * A takes his 1. 
  * B takes his 1. 



Hence, A's score is 1, B's score is 1 and difference is 0.

There is also another optimal way of playing:

  * A removes 5 from B's list. 
  * B removes 4 from A's list. 
  * A removes 1 from B's list. 
  * B removes 1 from A's list. 



The difference in the scores is still 0.

In the second example, irrespective of the moves the players make, they will end up with the same number of numbers added to their score, so the difference will be 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())

a = map(int,raw_input().split())
b = map(int,raw_input().split())

t = []
for i in range(n):
	t.append((a[i],1))
	t.append((b[i],2))
t.sort()
s1 = 0
s2 = 0
# print t
for i in range(1,2*n+1):
	if t[2*n-i][1] == 1 and i%2 == 1:
		s1 += t[2*n-i][0]
	if t[2*n-i][1] == 2 and i%2 == 0:
		s2 += t[2*n-i][0]
	# print s1,s2
print s1-s2
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cgamblingbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=5, max_value=10**6):
        self.n_min = n_min
        self.n_max = n_max
        self.max_value = max_value
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        a = [random.randint(1, self.max_value) for _ in range(n)]
        b = [random.randint(1, self.max_value) for _ in range(n)]
        return {'n': n, 'a': a, 'b': b}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        a = question_case['a']
        b = question_case['b']
        problem = (
            "Two players A and B have lists of integers and take turns making optimal moves to maximize their score difference (A's score minus B's).\n\n"
            "Game Rules:\n"
            "1. Player A starts first. The game ends when both lists are empty.\n"
            "2. On a turn, a player can either:\n"
            "   - Take an element from their own list (adds to their score, element is removed)\n"
            "   - Remove an element from the opponent's list\n"
            "3. Both players play optimally to maximize their own advantage.\n\n"
            "Input Details:\n"
            f"- First line: n = {n} (size of each list)\n"
            f"- Second line (A's list): {', '.join(map(str, a))}\n"
            f"- Third line (B's list): {', '.join(map(str, b))}\n\n"
            "Compute the final score difference (A - B). Put your answer within [answer] and [/answer], e.g., [answer]0[/answer]."
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            correct = cls.calculate_correct_answer(identity)
            return int(solution) == correct
        except:
            return False
    
    @classmethod
    def calculate_correct_answer(cls, identity):
        n = identity['n']
        a = identity['a']
        b = identity['b']
        merged = []
        for x in a:
            merged.append((x, 1))
        for x in b:
            merged.append((x, 2))
        merged.sort()
        s1, s2 = 0, 0
        for i in range(1, 2 * n + 1):
            val, player = merged[2 * n - i]
            if i % 2 == 1:  # A's turn
                if player == 1:
                    s1 += val
            else:  # B's turn
                if player == 2:
                    s2 += val
        return s1 - s2
