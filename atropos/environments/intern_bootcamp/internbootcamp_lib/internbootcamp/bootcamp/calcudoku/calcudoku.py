import re
import ast
import json
import sys
sys.path.append('./')
from internbootcamp.bootcamp.base import Basebootcamp
# from environments import CalcudokuEnvironment
from internbootcamp.libs.calcudoku.calcudoku_generator import CalcudokuGenerator
import random

class Calcudokubootcamp(Basebootcamp):
    
    def __init__(self, size:int = 6,  group_size_range:tuple =(1,4)):
        super().__init__()   
        self.size = size
        self.group_size_range = group_size_range
        # self.env = CalcudokuEnvironment()
    
    def generator(self, size:int =6, group_size_range:tuple =(1,4), seed:int = None):
        generator = CalcudokuGenerator(n=size, group_size_range=group_size_range, seed=seed)
        self.grid = generator.generate_puzzle()
        return self.grid

    def get_question(self):
        statements = [f"""You are an intelligent assistant specializing in solving custom puzzle problems. Below is a specific rule defined for a custom puzzle. Your task is to apply this rule accurately to the provided question.

    ### Instructions:
    
    1. Thoroughly understand the rule provided. If needed, break down the rule into simpler components or steps.
    2. Apply the rule carefully to address the question presented.
    3. Verify your answer to ensure it aligns with the rule and the context of the puzzle.
    
### Calcudoko Puzzle Rule:

1.Calcudoko is a sudoku-like game. The game is played on an NxN grid. In each row and column, fill in the numbers from 1 to N. 
2.As in sudoku, each number can appear only once in each row and column. 
3.The grid is divided into groups, each of which has a target number and an operator that indicates that the numbers in the group must equal the target number after a specified operation.
4.The operations :
    4.Sum: + The numbers in this group must add to the target value.
    4.Difference: - One of the numbers in this group, minus the rest, must equal the target value (the number to be subtracted from may be any of the cells).
    4.Product: * The numbers in this group must multiply to produce the target value.
    4.Ratio: / One of the numbers in this group, divided by all of the others, must equal the target value. As in a difference group, which of the cells contains the number to be divided is not specified.
5.Numbers may be repeated within a group (so long as they're not also repeated within a row or column).  
6.Puzzles are parsed in a "puzzle spec" format, where rows are given one per line, with cells separated by spaces. Groups are labelled with alphabetic characters, which are then used to identify cell membership. 

### Question

Now the "puzzle spec" of a Calcudoko is: 
{self.grid} 

The answer needs to provide the corresponding numbers for all positions in the Calcudoko.
""",
f"""
Calcudoko is a sudoku-like game. The game is played on an NxN grid. In each row and column, fill in the numbers from 1 to N. 
As in sudoku, each number can appear only once in each row and column. 
The grid is divided into groups, each of which has a target number and an operator that indicates that the numbers in the group must equal the target number after a specified operation.
Sum: +
The numbers in this group must add to the target value.
Difference: -
One of the numbers in this group, minus the rest, must equal the target value (the number to be subtracted from may be any of the cells).
Product: *
The numbers in this group must multiply to produce the target value.
Ratio: /
One of the numbers in this group, divided by all of the others, must equal the target value. As in a difference group, which of the cells contains the number to be divided is not specified.

Numbers may be repeated within a group (so long as they're not also repeated within a row or column).  
Puzzles are parsed in a "puzzle spec" format, where rows are given one per line, with cells separated by spaces. Groups are labelled with alphabetic characters, which are then used to identify cell membership. 

Now the "puzzle spec" of a Calcudoko is: 
{self.grid} 

The answer needs to provide the corresponding numbers for all positions in the Calcudoko.
"""]
        
        return statements[random.randint(0,len(statements)-1)]

    def case_generator(self):
        grid = self.generator(self.size, self.group_size_range)
        self.prompt = self.get_question()
        return self.parse_question(self.prompt)

    def prompt_func(self, identity) -> str:
        """
        Process the input_data and return the processed prompt.
        
        Args:
            question_ori: The question to be processed.
        
        Returns:
            str: The processed prompt.
        """
        # print("`identity` is ignored!!!!!")

        prompt = self.prompt + """\nThe output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets,like this: [[1 3 2,2 1 3,3 2 1]]. Making sure the size of your answer should be same as the size of the Calcudoko."""
        return prompt
        # instruction_following = """Let's think step by step and output the final answer with an example json formatting for a 5x5 board: 
        # Final-answer: ```json
        # {'A': [(row_a, col_a)],'B': [(row_b, col_b)],'C': [(row_c, col_c)],'D': [(row_d, col_d)],'E': [(row_e, col_e)]}
        # ```
        # """
        # prompt = question_ori + '\n' + instruction_following
        # return prompt

    @staticmethod
    def parse_question(question: str) -> dict:
        # 匹配谜题规格的数组部分
        match = re.search(r"\[(?:'[^']*'[,\s]*)*\]", question)
        if not match:
            return None
        array_str = match.group(0)
        try:
            puzzle_spec = ast.literal_eval(array_str)
        except:
            return None

        puzzle_rows = [row.split() for row in puzzle_spec]
        n = len(puzzle_rows)
        for row in puzzle_rows:
            if len(row) != n:
                return None

        groups: Dict[str, tuple] = {}
        puzzle_grid: List[List[str]] = []
        for row in puzzle_rows:
            grid_row = []
            for cell in row:
                group_char = cell[0]
                grid_row.append(group_char)
                # 提取运算符和目标值（如果有的话）
                op_match = re.fullmatch(r'^[A-Za-z]([+*/-])(\d+)$', cell)
                if op_match and group_char not in groups:
                    op = op_match.group(1)
                    target = int(op_match.group(2))
                    groups[group_char] = (op, target)
            puzzle_grid.append(grid_row)
        
        return {
            'groups': groups,
            'grid': puzzle_grid,
            'size': n
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
        # 提取双括号中的内容
        match = re.findall(r'\[\[(.*?)\]\]', response, re.DOTALL)
        if len(match) == 0:
            return None
        content = match[-1]
        rows = [row.strip('[] ') for row in content.split(',')]
        solution = []
        for row in rows:
            try:
                numbers = list(map(int, row.strip().split()))
            except:
                return None
            solution.append(numbers)
        return solution
        

    @staticmethod
    def check_solution(parsed_question: dict, parsed_response: dict) -> bool:
        n = parsed_question['size']
        grid = parsed_question['grid']
        groups = parsed_question['groups']
        solution = parsed_response
        
        # 检查行和列的有效性
        if len(solution) != n or any(len(row) != n for row in solution):
            return False
        for i in range(n):
            if sorted(solution[i]) != list(range(1, n+1)):
                return False
            col = [solution[j][i] for j in range(n)]
            if sorted(col) != list(range(1, n+1)):
                return False
        
        # 构建分组数字映射
        group_numbers = {}
        for i in range(n):
            for j in range(n):
                group = grid[i][j]
                num = solution[i][j]
                if group not in group_numbers:
                    group_numbers[group] = []
                group_numbers[group].append(num)
        
        # 验证每个分组
        for group, info in groups.items():
            nums = group_numbers.get(group, [])
            op, target = info
            
            if op == '+':
                if sum(nums) != target:
                    return False
            elif op == '*':
                product = 1
                for num in nums:
                    product *= num
                if product != target:
                    return False
            elif op == '-':
                total = sum(nums)
                if not any(2*x == total + target for x in nums):
                    return False
            elif op == '/':
                for x in nums:
                    product = 1
                    for num in nums:
                        if num != x:
                            product *= num
                    if product != 0 and x / product == target:
                        break
                else:
                    return False
            else:
                return False
        
        return True

    @classmethod
    def _verify_correction(cls, solution, identity):
        return cls.check_solution(identity, solution)