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
x, y = map(int, raw_input().split())

def gcd(x,y):
	if y > x:
		x = x^y; y = x^y; x = x^y
	if x%y == 0:
		return y
	return gcd(y, x%y)

ch = ['A', 'B']
s = []
b = 0
if x < y:
	b = 1
	x = x^y; y = x^y; x = x^y
if gcd(x,y) > 1:
	print 'Impossible'
else:
	while y!=0:
		l = x//y
		s.append(l)
		r = x%y
		x = y
		y = r
	s[-1]-=1
	st = ''
	for el in s:	
		st += '{}{}'.format(el, ch[b])
		b = 1-b
	print st
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import re
import random
from bootcamp import Basebootcamp

class Ealiceboborangesandapplesbootcamp(Basebootcamp):
    def __init__(self, max_value=10**6):
        self.max_value = max_value
    
    def case_generator(self):
        while True:
            x = random.randint(1, self.max_value)
            y = random.randint(1, self.max_value)
            if x * y > 1:
                if math.gcd(x, y) == 1:  # 确保合法实例与非法实例均衡生成
                    if random.choice([True, False]):
                        return {'x': x, 'y': y}
                else:
                    if random.choice([True, False]):
                        return {'x': x, 'y': y}

    @staticmethod
    def prompt_func(question_case) -> str:
        x = question_case['x']
        y = question_case['y']
        prompt = f"""Alice和Bob发现袋子里有{x}个橙子和{y}个苹果。Alice拿了1个橙子，Bob拿了1个苹果。他们按卡片序列执行以下操作：
- A卡：Alice将所有水果给Bob，并重新从袋子拿取等量水果
- B卡：Bob将所有水果给Alice，并重新从袋子拿取等量水果
最终必须刚好拿完所有水果。请给出合法的压缩卡片序列（如3B）或输出Impossible。

答案请包裹在[answer][/answer]标签中。示例：[answer]1A1B[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        solution = answer_blocks[-1].strip()
        
        if solution.lower() == 'impossible':
            return 'Impossible'
        
        if re.fullmatch(r'(([1-9]\d*[AB])+)', solution):
            return solution
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        x, y = identity['x'], identity['y']
        
        if solution == 'Impossible':
            return math.gcd(x, y) > 1
        
        decompressed = cls.decompress(solution)
        if not decompressed:
            return False
        
        return cls.validate_sequence(decompressed, x, y)

    @classmethod
    def decompress(cls, compressed):
        if not re.fullmatch(r'([1-9]\d*[AB])+', compressed):
            return None
        
        parts = re.findall(r'([1-9]\d*)([AB])', compressed)
        return ''.join(c * int(n) for n, c in parts)

    @classmethod
    def validate_sequence(cls, sequence, x, y):
        bag_orange = x - 1
        bag_apple = y - 1
        a_orange, a_apple = 1, 0
        b_orange, b_apple = 0, 1

        for c in sequence:
            if c == 'A':
                transfer_orange = a_orange
                transfer_apple = a_apple
                new_a_orange = a_orange
                new_a_apple = a_apple
                if (transfer_orange > bag_orange) or (transfer_apple > bag_apple):
                    return False
                
                b_orange += transfer_orange
                b_apple += transfer_apple
                bag_orange -= transfer_orange
                bag_apple -= transfer_apple
                
                a_orange = new_a_orange
                a_apple = new_a_apple

            elif c == 'B':
                transfer_orange = b_orange
                transfer_apple = b_apple
                new_b_orange = b_orange
                new_b_apple = b_apple
                if (transfer_orange > bag_orange) or (transfer_apple > bag_apple):
                    return False
                
                a_orange += transfer_orange
                a_apple += transfer_apple
                bag_orange -= transfer_orange
                bag_apple -= transfer_apple
                
                b_orange = new_b_orange
                b_apple = new_b_apple

            else:
                return False

        return bag_orange == 0 and bag_apple == 0
