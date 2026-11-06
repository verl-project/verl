import re
import random
import math
from itertools import count
from internbootcamp.bootcamp.base import Basebootcamp

class Bbehbooleanexpressionsbootcamp(Basebootcamp):
    """
    bootcamp for the Boolean Expressions task.
    
    This task requires determining which of five boolean expressions evaluates to True,
    where the expressions contain logical operators (and, or, not), mathematical comparisons,
    and statements about world capitals.
    """
    
    def __init__(self, max_depth=5, num_choices=5):
        """
        Initialize the bootcamp with default parameters.
        
        Args:
            - max_depth: Maximum depth of the generated expressions
            - num_choices: Number of choices in each question (default 5)
        """
        super().__init__()
        self.max_depth = max_depth
        self.num_choices = num_choices
        
        # Dictionary of facts about world capitals (country: capital)
        self.capital_facts = {
            'Afghanistan': 'Kabul',
            'Armenia': 'Yerevan',
            'Azerbaijan': 'Baku',
            'Belarus': 'Minsk',
            'Cameroon': 'Yaounde',
            'Canada': 'Ottawa',
            'Colombia': 'Bogota',
            'Denmark': 'Copenhagen',
            'Gambia': 'Banjul',
            'Germany': 'Berlin',
            'India': 'New Delhi',
            'Iran': 'Tehran',
            'Iraq': 'Baghdad',
            'Jordan': 'Amman',
            'Malaysia': 'Kuala Lumpur',
            'Nepal': 'Kathmandu',
            'Nigeria': 'Abuja',
            'Norway': 'Oslo',
            'Turkey': 'Ankara'
        }
        
        # Also include incorrect capitals for false statements
        self.false_capitals = {
            'Afghanistan': ['Kandahar', 'Herat'],
            'Armenia': ['Gyumri', 'Vanadzor'],
            'Azerbaijan': ['Ganja', 'Sumqayit'],
            'Belarus': ['Grodno', 'Brest'],
            'Cameroon': ['Douala', 'Garoua'],
            'Canada': ['Toronto', 'Vancouver'],
            'Colombia': ['Medellin', 'Cali'],
            'Denmark': ['Aarhus', 'Odense'],
            'Gambia': ['Libreville', 'Serrekunda'],
            'Germany': ['Munich', 'Hamburg'],
            'India': ['Mumbai', 'Delhi'],
            'Iran': ['Isfahan', 'Shiraz'],
            'Iraq': ['Basra', 'Mosul'],
            'Jordan': ['Beirut', 'Zarqa'],
            'Malaysia': ['Putrajaya', 'Johor Bahru'],
            'Nepal': ['Pokhara', 'Bhaktapur'],
            'Nigeria': ['Lagos', 'Kano'],
            'Norway': ['Bergen', 'Trondheim'],
            'Turkey': ['Istanbul', 'Izmir']
        }
    
    def _generate_math_expression(self):
        """Generate a mathematical expression that evaluates to True or False."""
        operators = ['greater than', 'is less than or equal to', 'is greater than', '<=', '>']
        
        # Generate random numbers
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        d = random.randint(-10, 10)
        
        # Randomly choose expression type
        expr_type = random.choice([
            f"{a} * {b} + {c} * {d} is less than or equal to {random.randint(-10, 10)} * {random.randint(-10, 10)}",
            f"{a} * {b} > {c}",
            f"{a} - ({b} / {c}) is greater than {d}",
            f"{a} - ({b} / {c}) <= {d}",
            f"max({a}, {b}, {c}, {d}) - min({a}, {b}, {c}, {d}) is greater than {random.randint(1, 10)}",
            f"max({a}, {b}, {c}, {d}) - min({a}, {b}, {c}, {d}) <= {random.randint(1, 10)}"
        ])
        
        return expr_type
    
    def _generate_capital_fact(self, is_true=True):
        """Generate a statement about a country's capital that is either true or false."""
        country = random.choice(list(self.capital_facts.keys()))
        
        if is_true:
            return f"The capital of {country} is {self.capital_facts[country]}."
        else:
            false_capital = random.choice(self.false_capitals[country])
            return f"The capital of {country} is {false_capital}."
    
    def _generate_atomic_expr(self, target_value=None):
        """Generate a simple atomic expression with a specified truth value."""
        if target_value is None:
            # If no target value specified, randomly choose True or False
            return random.choice([True, False])
        
        if target_value:
            # Generate a True statement
            expr_type = random.choice(['bool', 'capital', 'math'])
            if expr_type == 'bool':
                return "True"
            elif expr_type == 'capital':
                return self._generate_capital_fact(is_true=True)
            else:
                # Create a mathematical expression that evaluates to True
                # This requires more careful construction to ensure it's True
                a = random.randint(1, 10)
                return f"{a} * {a} > {a}"
        else:
            # Generate a False statement
            expr_type = random.choice(['bool', 'capital', 'math'])
            if expr_type == 'bool':
                return "False"
            elif expr_type == 'capital':
                return self._generate_capital_fact(is_true=False)
            else:
                # Create a mathematical expression that evaluates to False
                a = random.randint(1, 10)
                return f"{a} * {-a} > {a * a}"
    
    def _generate_expression(self, depth=0, target_value=None):
        """
        Generate a boolean expression with a target truth value.
        
        Args:
            depth: Current depth of the expression tree
            target_value: Desired truth value of the expression
            
        Returns:
            tuple: (expression string, actual truth value)
        """
        if depth >= self.max_depth or random.random() < 0.3:
            # Base case: return an atomic expression
            if isinstance(target_value, bool):
                return self._generate_atomic_expr(target_value), target_value
            else:
                val = bool(random.getrandbits(1))
                return self._generate_atomic_expr(val), val
        
        # Choose an operator
        operator = random.choice(['and', 'or', 'not'])
        
        if operator == 'not':
            # For NOT, we need the opposite of our target value
            if isinstance(target_value, bool):
                sub_target = not target_value
            else:
                sub_target = bool(random.getrandbits(1))
            
            sub_expr, sub_val = self._generate_expression(depth + 1, sub_target)
            expr = f"not ({sub_expr})"
            return expr, not sub_val
        
        else:  # 'and' or 'or'
            if operator == 'and':
                # For AND to be True, both operands must be True
                if target_value:
                    left_target = True
                    right_target = True
                else:
                    # For AND to be False, at least one operand must be False
                    if random.random() < 0.5:
                        left_target = False
                        right_target = random.choice([True, False])
                    else:
                        left_target = random.choice([True, False])
                        right_target = False
            else:  # 'or'
                # For OR to be True, at least one operand must be True
                if target_value:
                    if random.random() < 0.5:
                        left_target = True
                        right_target = random.choice([True, False])
                    else:
                        left_target = random.choice([True, False])
                        right_target = True
                else:
                    # For OR to be False, both operands must be False
                    left_target = False
                    right_target = False
            
            left_expr, left_val = self._generate_expression(depth + 1, left_target)
            right_expr, right_val = self._generate_expression(depth + 1, right_target)
            
            if operator == 'and':
                result_val = left_val and right_val
            else:
                result_val = left_val or right_val
            
            expr = f"({left_expr}) {operator} ({right_expr})"
            return expr, result_val
    
    def _generate_complex_expression(self, target_value):
        """Generate a more complex boolean expression for the puzzle."""
        # Start with a simple expression
        expr, actual_val = self._generate_expression(0, target_value)
        
        # Add more complexity with nested not operations
        if random.random() < 0.7:
            not_count = random.randint(1, 3)
            for _ in range(not_count):
                expr = f"not not {expr}"
                # Double negation doesn't change the value
        
        return expr, actual_val
    
    def case_generator(self):
        """
        Generate a boolean expressions puzzle case.
        
        Returns:
            dict: A dictionary containing the puzzle data including:
                - expressions: List of expressions
                - answer: The index of the expression that evaluates to True
        """
        expressions = []
        answer = random.randint(0, self.num_choices - 1)
        
        # Generate expressions
        for i in range(self.num_choices):
            target_value = (i == answer)  # True only for the answer
            expr, actual_val = self._generate_complex_expression(target_value)
            
            # Validate that we got what we wanted
            if actual_val != target_value:
                # Try again if necessary
                attempts = 0
                while actual_val != target_value and attempts < 5:
                    expr, actual_val = self._generate_complex_expression(target_value)
                    attempts += 1
            
            expressions.append(expr)
        
        return {
            "expressions": expressions,
            "answer": answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        Convert the generated case into a formatted prompt for the model.
        
        Args:
            question_case: The case generated by case_generator
            
        Returns:
            str: Formatted prompt for the model
        """
        expressions = question_case["expressions"]
        
        # Create the prompt
        prompt = "From the following five expressions, only one evaluates to true. Which one is it?\n\n"
        
        # Add each expression as a choice
        for i, expr in enumerate(expressions):
            choice = chr(65 + i)  # A, B, C, D, E
            prompt += f"({choice}) {expr}\n"
        
        # Add instructions for the answer format
        prompt += "\nEvaluate each expression and determine which one is true. Provide your answer as a single letter (A, B, C, D, or E) within [answer] tags. For example: [answer]C[/answer]"
        
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        Extract the answer from the model's response.
        
        Args:
            output: The model's complete response
            
        Returns:
            str: The extracted answer (A, B, C, D, or E) or None if not found
        """
        # Look for the answer in the [answer] tags
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        
        if not matches:
            return None
        
        # Get the last match (in case there are multiple)
        last_match = matches[-1].strip()
        
        # Normalize to just the letter
        if last_match and len(last_match) >= 1:
            answer = last_match[0].upper()
            
            # Validate the answer is one of the valid options
            if answer in ['A', 'B', 'C', 'D', 'E']:
                return answer
        
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity) -> bool:
        """
        Verify if the provided solution is correct.
        
        Args:
            solution: The extracted solution (A, B, C, D, or E)
            identity: The puzzle case dictionary
            
        Returns:
            bool: True if the solution is correct, False otherwise
        """
        # Convert the solution letter to the index
        if solution is None:
            return False
        
        solution_index = ord(solution) - ord('A')
        
        # Check if the solution matches the correct answer
        return solution_index == identity["answer"]
    
if __name__ == "__main__":
    # 初始化训练场
    bootcamp = BBEHBooleanExpressionsbootcamp()
    
    # 生成谜题实例
    case = bootcamp.case_generator()

    # 将谜题转换为文本问题
    prompt = BBEHBooleanExpressionsbootcamp.prompt_func(case)
    
    # 打印问题
    print("问题:")
    print(prompt)
    print("\n预期答案:", case["answer"])

