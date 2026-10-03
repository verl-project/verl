"""# 

### 谜题描述
Alice and Bob decided to eat some fruit. In the kitchen they found a large bag of oranges and apples. Alice immediately took an orange for herself, Bob took an apple. To make the process of sharing the remaining fruit more fun, the friends decided to play a game. They put multiple cards and on each one they wrote a letter, either 'A', or the letter 'B'. Then they began to remove the cards one by one from left to right, every time they removed a card with the letter 'A', Alice gave Bob all the fruits she had at that moment and took out of the bag as many apples and as many oranges as she had before. Thus the number of oranges and apples Alice had, did not change. If the card had written letter 'B', then Bob did the same, that is, he gave Alice all the fruit that he had, and took from the bag the same set of fruit. After the last card way removed, all the fruit in the bag were over.

You know how many oranges and apples was in the bag at first. Your task is to find any sequence of cards that Alice and Bob could have played with.

Input

The first line of the input contains two integers, x, y (1 ≤ x, y ≤ 1018, xy > 1) — the number of oranges and apples that were initially in the bag.

Output

Print any sequence of cards that would meet the problem conditions as a compressed string of characters 'A' and 'B. That means that you need to replace the segments of identical consecutive characters by the number of repetitions of the characters and the actual character. For example, string AAABAABBB should be replaced by string 3A1B2A3B, but cannot be replaced by 2A1A1B2A3B or by 3AB2A3B. See the samples for clarifications of the output format. The string that you print should consist of at most 106 characters. It is guaranteed that if the answer exists, its compressed representation exists, consisting of at most 106 characters. If there are several possible answers, you are allowed to print any of them.

If the sequence of cards that meet the problem statement does not not exist, print a single word Impossible.

Examples

Input

1 4


Output

3B


Input

2 2


Output

Impossible


Input

3 2


Output

1A1B

Note

In the first sample, if the row contained three cards with letter 'B', then Bob should give one apple to Alice three times. So, in the end of the game Alice has one orange and three apples, and Bob has one apple, in total it is one orange and four apples.

In second sample, there is no answer since one card is not enough for game to finish, and two cards will produce at least three apples or three oranges.

In the third sample, cards contain letters 'AB', so after removing the first card Bob has one orange and one apple, and after removal of second card Alice has two oranges and one apple. So, in total it is three oranges and two apples.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/python

import sys

def vec(a, b):
	return a[0] * b[1] - a[1] * b[0]


x, y = map(int, raw_input().split())

a = [1, 0]
b = [0, 1]

v = [x, y]
ans = \"\"

while a[0] + b[0] <= x and a[1] + b[1] <= y:
	q = vec(a, v)
	w = abs(vec(b, v))
	if q < w:
		c = (w - 1) // q
		b = [b[0] + c * a[0], b[1] + c * a[1]]
		ans += str(c) + 'A'
	elif q > w:
		c = (q - 1) // w
		a = [a[0] + c * b[0], a[1] + c * b[1]]
		ans += str(c) + 'B'
	else:
		print ans if a[0] + b[0] == x and a[1] + b[1] == y else \"Impossible\"
		sys.exit(0)

print ans if a[0] + b[0] == x and a[1] + b[1] == y else \"Impossible\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Caliceboborangesandapplesbootcamp(Basebootcamp):
    def __init__(self, max_value=10**3, probability_impossible=0.3, seed=None):
        super().__init__()
        self.max_value = max_value
        self.probability_impossible = probability_impossible
        self.rng = random.Random(seed)
    
    @staticmethod
    def vec(a, b):
        return a[0] * b[1] - a[1] * b[0]

    @classmethod
    def check_solution_exists(cls, x, y):
        a = [1, 0]
        b = [0, 1]
        while True:
            sum_a = a[0] + b[0]
            sum_b = a[1] + b[1]
            if sum_a > x or sum_b > y:
                break
            v = [x, y]
            q = cls.vec(a, v)
            w = abs(cls.vec(b, v))
            if q < w:
                c = (w - 1) // q
                b = [b[0] + c * a[0], b[1] + c * a[1]]
            elif q > w:
                c = (q - 1) // w
                a = [a[0] + c * b[0], a[1] + c * b[1]]
            else:
                return sum_a == x and sum_b == y
        return a[0] + b[0] == x and a[1] + b[1] == y

    def case_generator(self):
        for _ in range(100):
            generate_impossible = self.rng.random() < self.probability_impossible
            x = self.rng.randint(1, self.max_value)
            y = self.rng.randint(1, self.max_value)
            if x * y <= 1:
                continue
            possible = self.check_solution_exists(x, y)
            if generate_impossible:
                if not possible:
                    return {'x': x, 'y': y}
            else:
                if possible:
                    return {'x': x, 'y': y}
        return {'x': 3, 'y': 2}
    
    @staticmethod
    def prompt_func(question_case):
        x = question_case['x']
        y = question_case['y']
        prompt = f"""Alice and Bob discovered {x} oranges and {y} apples. They each took one fruit and created a card game to distribute the rest. 

Rules:
- Card 'A': Alice gives all her fruits to Bob and replaces them from the bag
- Card 'B': Bob gives all his fruits to Alice and replaces them from the bag
- After processing all cards, the bag must be empty

Find a valid card sequence (compressed format like 3A1B) or 'Impossible'. Format your answer within [answer][/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        impossible_match = re.search(r'\bImpossible\b', output, re.IGNORECASE)
        if impossible_match:
            return 'Impossible'
        candidates = re.findall(r'(?:\d+[AB])+', output)
        if not candidates:
            return None
        last_candidate = candidates[-1]
        if Caliceboborangesandapplesbootcamp.decompress_solution(last_candidate) is None:
            return None
        return last_candidate
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        x_val = identity['x']
        y_val = identity['y']
        if solution == 'Impossible':
            return not cls.check_solution_exists(x_val, y_val)
        
        decompressed = cls.decompress_solution(solution)
        if decompressed is None:
            return False
        
        alice_orange, alice_apple = 1, 0
        bob_orange, bob_apple = 0, 1
        remaining_oranges = x_val - 1
        remaining_apples = y_val - 1
        
        for c in decompressed:
            if c == 'A':
                needed_o = alice_orange
                needed_a = alice_apple
            elif c == 'B':
                needed_o = bob_orange
                needed_a = bob_apple
            else:
                return False
            
            if remaining_oranges < needed_o or remaining_apples < needed_a:
                return False
            
            remaining_oranges -= needed_o
            remaining_apples -= needed_a
            
            if c == 'A':
                bob_orange += alice_orange
                bob_apple += alice_apple
            else:
                alice_orange += bob_orange
                alice_apple += bob_apple
        
        return (alice_orange + bob_orange == x_val and 
                alice_apple + bob_apple == y_val and
                remaining_oranges == 0 and 
                remaining_apples == 0)
    
    @staticmethod
    def decompress_solution(solution):
        if solution == 'Impossible':
            return solution
        parts = re.findall(r'(\d+)([AB])', solution)
        if not parts:
            return None
        decompressed = []
        for cnt, c in parts:
            if not cnt.isdigit() or cnt.startswith('0'):
                return None
            count = int(cnt)
            if count < 1:
                return None
            decompressed.append(c * count)
        return ''.join(decompressed) if decompressed else None
