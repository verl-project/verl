import re
import json
from internbootcamp.bootcamp.base import Basebootcamp
from internbootcamp.libs.game24.game24 import Game24Plus

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

class Game24bootcamp(Basebootcamp):
    def __init__(self, num_numbers=4, range_max=100, target_max=100, seed = None):
        # super.__init__()
        self.num_numbers = num_numbers
        self.range_max = range_max
        self.target_max = target_max
        self.seed = seed
        self.game24Plus = Game24Plus(num_numbers,range_max,target_max,seed)
        
    def case_generator(self) -> object:
        """
        生成一组数字和目标值。
        """
        # game24Plus = Game24Plus(num_numbers,range_max,target_max,seed)
        self.numbers = self.game24Plus.get_numbers()
        self.target, self.operations = self.game24Plus.get_target_limit_range(self.numbers)
        return {'puzzle': ' '.join(str(i) for i in self.numbers), 'target':self.target}
    
    def prompt_func(self, identity) -> str:
        """
        Process the input_data and return the processed prompt.
        
        Args:
            question_ori: The question to be processed.
        
        Returns:
            str: The processed prompt.
        """
        instruction = f"请解决以下问题：使用数字 {identity['puzzle']} 通过加减乘除得到 {identity['target']}。"
        instruction_following = """Let's think step by step and output the final answer within \\boxed{}.The final answer should be all input numbers with basic operations, and parentheses can be used to change the order of operations. For example "Final Answer: \\boxed{6+6+(6+6)}"."""
        
        prompt = instruction + '\n' + instruction_following
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        Extract the output from the solution.
        
        Args:
            output: Model output to be processed.
        
        Returns:
            The processed output.
        """
        output = last_boxed_only_string(output)
        if output is None:
            return None
        return remove_boxed(output)
    
    @staticmethod
    def _normalize_solution(solution):
        """
        Normalize the solution.
        """
        solution = solution.replace("**", "")
        solution = solution.replace("\\times", "*")
        solution = solution.replace("\\div", "/")
        solution = solution.replace("\\left", "")
        solution = solution.replace("\\right", "")
        return solution
        
    @classmethod 
    def _verify_correction(cls, solution, identity)->bool:
        """
        Verify the correction of the solution.
        """ 
        puzzle, target = identity['puzzle'], identity['target']
        current_numbers = puzzle.split()
        solution = cls._normalize_solution(solution)
        # 提取所有数字
        numbers = re.findall(r'\d+', solution)
        # 提取所有运算符
        operators = re.findall(r'[+\-*/]', solution)
        
        # 检查数字
        if len(numbers) != len(current_numbers):
            return False
        for n in numbers:
            if n not in current_numbers:
                return False
            current_numbers.remove(n)
        if current_numbers:
            return False
        # 检查运算符
        if '**' in solution or '//' in solution:
            return False
        if any(op not in '+-*/' for op in operators):
            return False
        
        try:
            # 计算结果
            result = eval(solution)
            if result != int(target):
                return False
        except:
            return False
        
        return True
        
        



if __name__ == '__main__':
    bootcamp = Game24bootcamp
    data = [
        {
            "puzzle": "6 6 6 6",
            "target": "24",
            "solution": "\\boxed{6*6-6/6}",
        },
        {
            "puzzle": "5 9 10 20",
            "target": "30",
            "solution": "\\boxed{5*10-9+20}",
        },
        {
            "puzzle": "5 9 10 11",
            "target": "30",
            "solution": "\\boxed{5 * 10-9-11}",
        },
        {
            "puzzle": "5 9 10 11",
            "target": "30",
            "solution": "\\boxed{5 \\times 10-(9+11)}",
        },
        {
            "puzzle": "5 9 10 11",
            "target": "30",
            "solution": "\\boxed{5\\times10-9+11}",
        },
        {
            "puzzle": [0, 1, 3, 4],
            "target": "7",
            "solution": "\\boxed{(4 + 3) / (1 + 0)}",
        },

    ]
    
    for d in data:
        print(bootcamp.verify_score(d['solution'], d, short_threshold=1))
    