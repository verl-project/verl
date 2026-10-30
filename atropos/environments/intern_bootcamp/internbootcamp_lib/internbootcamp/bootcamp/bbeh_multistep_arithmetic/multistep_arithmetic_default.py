import json
import random
import math
from math import gcd
from typing import Dict, Any, List, Optional

from bootcamp import Basebootcamp
import random
import re
import json


import json
import random
import math
from math import gcd
from typing import Dict, Any, List, Optional
import re


def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


class BbehMultistepArithmeticbootcamp(Basebootcamp):
    def __init__(self, num_operators: int = 5, max_depth: int = 10, reuse_prob: float = 0.3,  **params):
        super().__init__(**params)
        self.num_operators = num_operators
        self.max_depth = max_depth
        self.reuse_prob = reuse_prob

    def case_generator(self) -> Dict:
        symbols = self._generate_operator_symbols(self.num_operators)
        operators = self._generate_operators(symbols)
        
        # 生成纯数字表达式
        A_expr = self._generate_expression(symbols, self.max_depth)
        B_expr = self._generate_expression(symbols, self.max_depth)
        C_expr = self._generate_expression(symbols, self.max_depth)
        
        # 计算表达式值
        op_map = {op['symbol']: self._create_operator_func(op) for op in operators}
        A_val = self._eval_expr(A_expr, op_map)
        B_val = self._eval_expr(B_expr, op_map)
        C_val = self._eval_expr(C_expr, op_map)
        
        return {
            'operators': operators,
            'A': A_expr,
            'B': B_expr,
            'C': C_expr,
            "A_val": A_val,
            "B_val": B_val,
            "C_val": C_val,
            'answer': A_val + B_val - C_val
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        operators = '\n'.join(
            [f'${op["symbol"]} b$ equals {op["true_expr"]} if {op["condition"]}; otherwise, {op["false_expr"]}'
             for op in question_case['operators']]
        )
        problem = (
            f"Consider the following new operations:\n\n{operators}\n"
            "For brevity, we use $a <op1><op2> b$ to denote $(a op1 b) op2 b$. For example, $4 +* -5$ means $(4 + -5) * -5$ and $4 *-- -5$ means $(4 * -5) -- -5$.\n"
            f"Let A = {question_case['A']}\n"
            f"Let B = {question_case['B']}\n"
            f"Let C = {question_case['C']}\n"
            "Compute A + B - C. Your final answer must be in number form. Please put your final answer within [answer] and [/answer] tags."
        )
        return problem

    @staticmethod
    def extract_output(output: str) -> Optional[float]:
        answers = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answers:
            return None
        try:
            return float(answers[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution: float, identity: Dict) -> bool:
        return abs(solution - identity['answer']) < 1e-6

    # Helper methods
    def _generate_operator_symbols(self, num: int) -> List[str]:
        candidates = ['><', ';', '][', '@', '#', '<>', '~', '&', '[]', ':*','!',]
        return random.sample(candidates, num)

    def _generate_operators(self, symbols: List[str]) -> List[Dict]:
        operators = []
        for i, symbol in enumerate(symbols):
            condition_type = random.choice([
                'product_positive', 'a_gt_b', 'prime_condition', 
                'gcd_condition', 'abs_diff'
            ])
            condition, true_expr, false_expr = self._generate_operator_def(
                condition_type, symbols[:i]
            )
            operators.append({
                'symbol': symbol,
                'condition': condition,
                'true_expr': true_expr,
                'false_expr': false_expr
            })
        return operators

    def _generate_operator_def(self, condition_type: str, available_symbols: List[str]) -> tuple:
        a, b = 'a', 'b'
        if condition_type == 'product_positive':
            cond = f"{a} * {b} > 0"
            true = f"{a} - {b}"
            false = f"{a} + {b}"
        elif condition_type == 'a_gt_b':
            cond = f"{a} > {b}"
            true = f"{a} * {b}"
            false = f"{a} - {b}" if random.random() < 0.5 else f"{a} + {b}"
        elif condition_type == 'prime_condition':
            cond = f"is_prime({a}) or is_prime({b})"
            true = f"min({a}, {b})"
            false = f"max({a}, {b})"
        elif condition_type == 'gcd_condition':
            cond = f"math.gcd({a}, {b}) == 1"
            true = f"{a} + {b}"
            false = f"math.gcd({a}, {b})"
        else:  # abs_diff
            cond = f"abs({a} - {b}) < 2"
            true = f"{a} * {b}"
            false = f"{a} - {b}"

        # 30%概率使用已有运算符
        if available_symbols and random.random() < self.reuse_prob:
            used_symbol = random.choice(available_symbols)
            true = f"({a} {used_symbol} {b})"
            false = f"({a} {used_symbol} {b})" if random.random() < 0.5 else false

        return cond, true, false

    def _generate_expression(self, symbols: List[str], depth: int) -> str:
        if depth == 0 or not symbols:
            return self._generate_operand()
        left = self._generate_expression(symbols, depth-1)
        right = self._generate_operand()
        composite = ''.join(random.choices(symbols, k=random.randint(1,2)))
        return f"({left} {composite} {right})"

    def _generate_operand(self) -> str:
        return str(random.choice([x for x in range(-10, 11) if x != 0]))

    def _create_operator_func(self, operator: Dict):
        condition = operator['condition']
        true_expr = operator['true_expr']
        false_expr = operator['false_expr']
        context = {
            'math': math,
            'self': self
        }
        def func(a, b):
            try:
                a = int(a) if isinstance(a, float) and a.is_integer() else a
                b = int(b) if isinstance(b, float) and b.is_integer() else b
                cond = eval(condition, {'a': a, 'b': b, **context})
                expr = true_expr if cond else false_expr
                return eval(expr, {'a': a, 'b': b, **context})
            except:
                return 0
        return func

    def _eval_expr(self, expr: str, op_map: Dict) -> float:
        expr = expr.replace(' ', '')
        sorted_ops = sorted(op_map.keys(), key=lambda x: -len(x))
        
        def parse_operators(s):
            ops = []
            i = 0
            while i < len(s):
                for op in sorted_ops:
                    if s.startswith(op, i):
                        ops.append(op)
                        i += len(op)
                        break
                else:
                    i += 1
            return ops
        
        def evaluate(s):
            if not s:
                return 0
            # 处理括号
            if s[0] == '(':
                balance = 1
                i = 1
                while i < len(s) and balance > 0:
                    if s[i] == '(': balance += 1
                    elif s[i] == ')': balance -= 1
                    i += 1
                inner_val = evaluate(s[1:i-1])
                remaining = s[i:]
            else:
                # 提取数字
                match = re.match(r'^([+-]?\d+)(.*)', s)
                if not match:
                    return 0
                inner_val = float(match.group(1))
                remaining = match.group(2)
            
            # 处理复合运算符
            while remaining:
                ops = parse_operators(remaining)
                if not ops:
                    break
                op_len = sum(len(op) for op in ops)
                remaining = remaining[op_len:]
                
                # 提取右操作数
                if not remaining:
                    right = 0
                elif remaining[0] == '(':
                    balance = 1
                    i = 1
                    while i < len(remaining) and balance > 0:
                        if remaining[i] == '(': balance += 1
                        elif remaining[i] == ')': balance -= 1
                        i += 1
                    right = evaluate(remaining[1:i-1])
                    remaining = remaining[i:]
                else:
                    match = re.match(r'^([+-]?\d+)(.*)', remaining)
                    if not match:
                        right = 0
                        remaining = ''
                    else:
                        right = float(match.group(1))
                        remaining = match.group(2)
                
                # 应用运算
                for op in ops:
                    if op in op_map:
                        inner_val = op_map[op](inner_val, right)
                    else:
                        inner_val = 0
            return inner_val
        
        return evaluate(expr)
    
    

if __name__ == "__main__":
    bootcamp = BbehMultiStepArithmeticbootcamp(num_operators = 5, max_depth = 10, reuse_prob = 0.3)
    case = bootcamp.case_generator()
    print(json.dumps(case, indent=2))
    ans = case['answer']
    prompt = bootcamp.prompt_func(case)
    print(prompt)
    true_response = f"[answer]{ans}[/answer]"
    false_response = f"[answer]{ans+1}[/answer]"
    correction = bootcamp._verify_correction(ans, case)
    print(correction)
    wrong_correction = bootcamp._verify_correction(ans+1, case)
    print(wrong_correction)