import re
import ast
import json
import sys
sys.path.append('./')
from internbootcamp.bootcamp.base import Basebootcamp
import random
from internbootcamp.libs.cryptomath.crypto_math import generate_crypto_math


class Cryptomathbootcamp(Basebootcamp):
    
    def __init__(self, num_letters=5, num_add=4, *args, **kwargs):
        # description = "略"
        # super().__init__(description,*args, **kwargs)
        self.num_letters = num_letters
        self.num_add = num_add
        # self.env = CryptoMathEnvironment(num_letters=num_letters, num_add=num_add)

    def generator(self):
        results = generate_crypto_math(self.num_letters, 1, self.num_add)
        puzzle = results[0]["puzzle"]
        self.puzzle = puzzle
        return self.puzzle
    def case_generator(self):
        puzzle = self.generator()
        self.prompt = self.get_question()
        self.prompt += self.get_question_following()

        return self.parse_question(self.prompt)

    def get_question(self):
        statements = [f"""你是一个专门解决定制谜题问题的智能助手。以下是为定制谜题定义的具体规则。你的任务是准确地将此规则应用到提供的问题上。

### 指示：

1. 彻底理解所提供的规则。如有必要，可以将规则分解成更简单的组成部分或步骤。
2. 小心地应用规则来解答所提出的问题。
3. 核对你的答案以确保它与规则和谜题的背景一致。

### 谜题规则：

1. 游戏给出一个字母公式，每个字母代表一个独一无二的数字（0-9）。
2. 不同的字母不能代表相同的数字。
3. 任何多位数的第一个字母不能代表0。

### 问题：
{self.puzzle}
请以字母=数字的形式提供你的答案，并确保将你的答案用双括号括起来，如下所示：[[A=1,B=2,...]]。

### 答案：""",
f"""你是一位专注于解决个性化谜题的智能助手。下面提供了一个特别为定制谜题设定的规则。你需要做的是，精准地运用这个规则来解答所提供的题目。

### 操作指南：

1. 仔细研读并理解规则内容。如果有必要，可以将规则拆解成更简单的部分或步骤来进行理解。
2. 精确地依照规则来处理给出的问题。
3. 检查你的答案，确保它与规则以及谜题的情境相符合。

### 谜题规则说明：

1. 游戏中会给出一个由字母组成的公式，其中每个字母都代表一个独一无二的数字（0-9）。
2. 不同的字母不能表示相同的数字。
3. 对于多位数而言，最左边的首位字母不能是0。

### 题目：
{self.puzzle}
请按照字母对应数字的方式给出你的答案，并且要将答案用双括号包围起来，例如：[[A=1,B=2,...]]。

### 解答：""",
f"""你是一个专长于解决定制谜题的智能助手。下面提供了一个用于特定谜题的规则，你需要准确地把这个规则应用到给出的问题上。

首先，你要完全理解所提供的规则，必要时可以将规则分解成简单的部分或步骤来帮助理解。然后，仔细地使用这个规则来解决问题。最后，检查你的答案，确保它符合规则和谜题的要求。

游戏会给出一个字母组成的公式，每个字母代表一个独一无二的数字（0-9），不同的字母不能代表相同的数字，并且任何多位数的首字母不能是0。

问题是：{self.puzzle}
""",
f"""Crypto Math是一种逻辑解谜游戏，其规则简单，解题过程富有挑战性。
游戏规则如下：游戏提供了一个字母组成的加法算式，每个字母代表一个0-9的整数数字，不同的字母不能表示相同的数字，任何多位数的第一个字母不能代表0。你需要推断出每个字母代表的数字，使得整个算式成立。
请完成该Crypto Math，输入算式为:\n {self.puzzle}"""]
        
        
        return random.choice(statements)
    
    def get_question_following(self):
        followings = []
        followings.append("""\n
请以字母等于数字的形式给出你的答案，并且把答案放在双括号内，比如这样：[[A=1,B=2,...]]。""")
        return random.choice(followings)

    def prompt_func(self, identity) -> str:
        return self.prompt

    @staticmethod
    def parse_question(question: str) -> dict:
        pattern = r'(?:问题|题目|输入算式为|问题是)[：:]\s*([A-Z+]+=[A-Z]+)'
        match = re.search(pattern, question)
        if not match:
            return None
        equation = match.group(1)
        left, right = equation.split('=')
        terms = left.split('+')
        leading_letters = set()
        letters = set()
        for term in terms + [right]:
            letters.update(term)
            if len(term) > 1:
                leading_letters.add(term[0])
        return {
            'left_terms': terms,
            'right_term': right,
            'leading_letters': list(leading_letters),
            'all_letters': list(letters)
        }

    @staticmethod
    def extract_output(response):
        """
        Extract the output from the solution.
        
        Args:
            output: Model output to be processed.
        
        Returns:
            The processed output.
        """
        # if re.search(r'\[\[No solution\]\]', response, re.IGNORECASE):
        #     return None
        content_match = re.findall(r'\[\[(.*?)\]\]', response)
        if len(content_match) == 0:
            return None
        content = content_match[-1].replace(' ', '')
        pairs = re.findall(r'([A-Z])=(\d+)', content)
        if not pairs:
            return None
        solution = {}
        for letter, num_str in pairs:
            if not num_str.isdigit():
                return None
            num = int(num_str)
            solution[letter] = num
        return solution
        

    @staticmethod
    def check_solution(parsed_question: dict, parsed_response: dict) -> bool:
        def has_solution(pq):
            letters = list(pq['all_letters'])
            leading = pq['leading_letters']
            n = len(letters)
            for perm in permutations(range(10), n):
                assignment = dict(zip(letters, perm))
                valid = all(assignment[l] != 0 for l in leading)
                if not valid:
                    continue
                left_sum = 0
                for term in pq['left_terms']:
                    num = 0
                    for c in term:
                        num = num * 10 + assignment[c]
                    left_sum += num
                right_num = 0
                for c in pq['right_term']:
                    right_num = right_num * 10 + assignment[c]
                if left_sum == right_num:
                    return True
            return False

        if parsed_response is None:
            return not has_solution(parsed_question)
        else:
            pq = parsed_question
            resp = parsed_response
            leading = pq['leading_letters']
            for letter in leading:
                if resp.get(letter, 0) == 0:
                    return False
            values = list(resp.values())
            if len(values) != len(set(values)):
                return False
            if set(resp.keys()) != set(pq['all_letters']):
                return False
            left_sum = 0
            for term in pq['left_terms']:
                num = 0
                for c in term:
                    num = num * 10 + resp[c]
                left_sum += num
            right_num = 0
            for c in pq['right_term']:
                right_num = right_num * 10 + resp[c]
            return left_sum == right_num


    @classmethod
    def _verify_correction(cls, solution, identity):
        return cls.check_solution(identity, solution)