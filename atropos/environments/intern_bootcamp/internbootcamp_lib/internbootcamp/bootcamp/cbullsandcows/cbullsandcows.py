"""# 

### 谜题描述
The \"Bulls and Cows\" game needs two people to play. The thinker thinks of a number and the guesser tries to guess it.

The thinker thinks of a four-digit number in the decimal system. All the digits in the number are different and the number may have a leading zero. It can't have more than one leading zero, because all it's digits should be different. The guesser tries to guess the number. He makes a series of guesses, trying experimental numbers and receives answers from the first person in the format \"x bulls y cows\". x represents the number of digits in the experimental number that occupy the same positions as in the sought number. y represents the number of digits of the experimental number that present in the sought number, but occupy different positions. Naturally, the experimental numbers, as well as the sought number, are represented by four-digit numbers where all digits are different and a leading zero can be present.

For example, let's suppose that the thinker thought of the number 0123. Then the guessers' experimental number 1263 will receive a reply \"1 bull 2 cows\" (3 occupies the same positions in both numbers and 1 and 2 are present in both numbers but they occupy different positions). Also, the answer to number 8103 will be \"2 bulls 1 cow\" (analogically, 1 and 3 occupy the same positions and 0 occupies a different one). 

When the guesser is answered \"4 bulls 0 cows\", the game is over.

Now the guesser has already made several guesses and wants to know whether his next guess can possibly be the last one.

Input

The first input line contains an integer n (1 ≤ n ≤ 10) which represents the number of already made guesses. Then follow n lines in the form of \"ai bi ci\", where ai is the i-th experimental number, bi is the number of bulls, ci is the number of cows (1 ≤ i ≤ n, 0 ≤ bi, ci, bi + ci ≤ 4). The experimental numbers are correct, i.e., each of them contains exactly four digits, in each of them all the four digits are different, and there can be a leading zero. All the experimental numbers are different. As the guesser hasn't guessed the number yet, the answer \"4 bulls 0 cows\" is not present.

Output

If the input data is enough to determine the sought number, print the number with four digits on a single line. If it has less than four digits, add leading zero. If the data is not enough, print \"Need more data\" without the quotes. If the thinker happens to have made a mistake in his replies, print \"Incorrect data\" without the quotes.

Examples

Input

2
1263 1 2
8103 2 1


Output

Need more data

Input

2
1234 2 2
1256 0 2


Output

2134

Input

2
0123 1 1
4567 1 2


Output

Incorrect data

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from itertools import permutations
def ch(ww):
    global s, kb
    t=0
    for i in range(n):
        b=0
        w=list(ww)
        u=list(s[i][0])
        for j in (3,2,1,0):
            if u[j]==w[j]:
                b+=1
                del u[j]
                del w[j]
        k=len(set(u)&set(w))
        if b==int(s[i][1]) and k==int(s[i][2]): t+=1
    if t==n:
        kb=ww
        return 1
    else: return 0

n = int(raw_input())
s=[[] for i in range(n)]
for k in range(n):
    s[k] =raw_input().split()

kb=''
st=0
for i in permutations('9876543210',4):
    st+=ch(''.join(i))
    if st>1:
        print ('Need more data')
        break
else:
    if st==1: print (kb)
    else: print ('Incorrect data')
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from itertools import permutations
from bootcamp import Basebootcamp

class Cbullsandcowsbootcamp(Basebootcamp):
    def __init__(self, n=2):
        self.n = n  # Number of guesses to generate

    def case_generator(self):
        # Generate a correct number with all unique digits
        digits = list('0123456789')
        correct_number = ''.join(random.sample(digits, 4))
        
        guesses = []
        for _ in range(self.n):
            # Generate a guess with unique digits
            guess = ''.join(random.sample(digits, 4))
            # Calculate bulls and cows
            bulls = sum(1 for i in range(4) if guess[i] == correct_number[i])
            common = len(set(guess) & set(correct_number))
            cows = common - bulls
            guesses.append({
                'number': guess,
                'bulls': bulls,
                'cows': cows
            })
        
        case = {
            'n': self.n,
            'guesses': guesses,
            'correct_number': correct_number
        }
        return case

    @staticmethod
    def prompt_func(question_case):
        prompt = "You need to solve a Bulls and Cows puzzle. Based on the given guesses and their results, determine the correct four-digit number. The rules are as follows:\n\n"
        prompt += "Each guess gives x bulls and y cows, where:\n"
        prompt += "- bulls represent the number of digits that are correct in both value and position.\n"
        prompt += "- cows represent the number of digits that are correct in value but incorrect in position.\n\n"
        prompt += "Known guesses and results:\n"
        for i, g in enumerate(question_case['guesses'], 1):
            prompt += f"{i}. Guess: {g['number']}, Result: {g['bulls']} bulls {g['cows']} cows\n"
        prompt += "\nDetermine the correct four-digit number, or state if more data is needed or if the data is incorrect.\n"
        prompt += "Please place your answer within [answer] tags. For example:\n"
        prompt += "- If the correct number is 1234, output [answer]1234[/answer]\n"
        prompt += "- If more data is needed, output [answer]Need more data[/answer]\n"
        prompt += "- If the data is incorrect, output [answer]Incorrect data[/answer]\n"
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if matches:
            return matches[-1].strip()
        else:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        guesses = identity['guesses']
        
        possible = []
        for num in permutations('0123456789', 4):
            num_str = ''.join(num)
            valid = True
            for g in guesses:
                guess_num = g['number']
                bulls = sum(1 for i in range(4) if guess_num[i] == num_str[i])
                common = len(set(guess_num) & set(num_str))
                cows = common - bulls
                if bulls != g['bulls'] or cows != g['cows']:
                    valid = False
                    break
            if valid:
                possible.append(num_str)
        
        if len(possible) == 0:
            return solution == 'Incorrect data'
        elif len(possible) == 1:
            return solution == possible[0]
        else:
            return solution == 'Need more data'

# 示例使用
if __name__ == "__main__":
    # 初始化训练场
    bootcamp = Cbullsandcowsbootcamp(n=2)
    
    # 生成谜题实例
    case = bootcamp.case_generator()
    
    # 生成提示问题
    prompt = Cbullsandcowsbootcamp.prompt_func(case)
    print("Generated Prompt:")
    print(prompt)
    
    # 假设模型输出
    # 这里使用正确的数作为示例
    model_output = f"[answer]{case['correct_number']}[/answer]"
    
    # 提取答案
    extracted_solution = Cbullsandcowsbootcamp.extract_output(model_output)
    print(f"Extracted Solution: {extracted_solution}")
    
    # 验证并评分
    score = Cbullsandcowsbootcamp.verify_score(model_output, case)
    print(f"Score: {score}")
