import ast
import json
import random
import re

from internbootcamp.bootcamp.base import Basebootcamp
from internbootcamp.libs.maze.maze_generator import generate_maze
from internbootcamp.libs.maze.maze_solver import solve_maze, is_path_exist
from internbootcamp.libs.maze.maze_validator import validate_maze_solution

class Mazebootcamp(Basebootcamp):
    def __init__(self, size=(6,6), start_pos=(0,0), end_pos=None, difficulty=None, seed=None):
        self.grid = None
        self.size = tuple(size)
        self.start_pos = tuple(start_pos)
        self.end_pos = tuple(end_pos) if end_pos else (size[0] - 1, size[1] - 1)
        self.solution_path = None
        self.difficulty = difficulty if difficulty else random.randint(1, 3)
        self.seed = seed
    
    
    def case_generator(self):
        """
        生成一个迷宫谜题

        参数:
            size: 迷宫大小，(行数, 列数)
            start_pos: 起点位置
            end_pos: 终点位置，如果为None则默认为右下角
            difficulty: 难度级别 (1-3)
            seed: 随机种子

        返回:
            grid: 生成的迷宫网格
        """
        rows, cols = self.size
        self.start_pos = self.start_pos
        self.difficulty = self.difficulty

        # 如果没有指定终点，则默认为右下角
        if self.end_pos is None:
            self.end_pos = (rows - 1, cols - 1)
        else:
            self.end_pos = self.end_pos

        # print(f"生成 {rows}x{cols} 迷宫，难度 {self.difficulty}...")

        # 调用迷宫生成函数
        self.grid = generate_maze(rows, cols, self.start_pos, self.end_pos, self.difficulty, self.seed)

        # 确保迷宫有解
        if not is_path_exist(self.grid, self.start_pos, self.end_pos):
            # print("生成的迷宫无解，重新生成...")
            return self.case_generator()
        else:
            # print("迷宫生成成功！")
            identity = {
                'grid': self.grid,
                'start_pos': self.start_pos,
                'end_pos': self.end_pos,
                'difficulty': self.difficulty,
                'seed': self.seed,
                'solution_path': solve_maze(self.grid, self.start_pos, self.end_pos)[0],
            }
            return identity
    
    @staticmethod
    def extract_output(output):
        """
        Extract the output from the solution.
        
        Args:
            output: Model output to be processed.
        
        Returns:
            The processed output.
        """
        pattern = pattern = r'```json\s*([\s\S]*?)\s*```'
        matches = re.findall(pattern, output)
        if matches:
            python_str = matches[-1]
            try:
                result_dict = ast.literal_eval(python_str.strip())
                return result_dict
            except Exception:
                return python_str
        else:
            return None
    @staticmethod
    def print_maze_str(identity: dict):
        """返回迷宫的字符串表示"""
        if identity['grid'] is None:
            return "没有迷宫可显示"

        result = "Maze:\n"
        for i, row in enumerate(identity['grid']):
            line = ""
            for j, cell in enumerate(row):
                if (i, j) == identity['start_pos']:
                    line += "S "  # 起点
                elif (i, j) == identity['end_pos']:
                    line += "E "  # 终点
                elif cell == 0:
                    line += "P "  # 通路
                else:
                    line += "W "  # 墙
            result += line + "\n"
        return result
    
    @classmethod
    def prompt_func(cls,identity: dict) -> str:
        statements = [
            f"""你是一个擅长解决迷宫问题的智能助手。以下是一个迷宫问题，请找出从起点(S)到终点(E)的路径。

迷宫规则：
1. 迷宫由通路(P)和墙壁(W)组成，只能沿着通路(P)移动。
2. 每次移动只能向上、下、左、右四个方向之一移动一格。
3. 不能穿过墙壁(W)或对角线移动。
4. 目标是找到从起点(S)到终点(E)的路径。

迷宫如下：

{cls.print_maze_str(identity)}

请给出从起点到终点的完整路径，以坐标序列的形式表示。坐标格式为(行,列)，从0开始计数。
例如：[(0, 0), (0, 1), (1, 1)]
""",
            f"""You are an intelligent assistant specializing in solving maze puzzles. Below is a maze puzzle that needs to be solved.

Maze Rules:
1. The maze consists of passages(P) and walls(W). You can only move along the passages(P).
2. Each move can only be in one of four directions: up, down, left, or right.
3. You cannot move through walls(W) or diagonally.
4. The goal is to find a path from the start (S) to the end (E).

The maze is as follows:

{cls.print_maze_str(identity)}

Please provide the complete path from start to end as a sequence of coordinates. Coordinates are in (row,column) format, starting from 0.
For example: [(0, 0), (0, 1), (1, 1)]
"""
        ]
        instruction_following = """\nLet's think step by step and output the final answer like this markdown formatting: 
Final-answer: ```json
[(start_row, start_col), (path_row1, path_col1), (path_row2, path_col2), (path_row3, path_col3), (path_row4, path_col4), (path_row5, path_col5), (end_row, end_col)]
```"""
        return statements[random.randint(0, len(statements) - 1)] + instruction_following
    
    @classmethod
    def _verify_correction(cls,solution,identity)->bool:
        return validate_maze_solution(identity['grid'], tuple(identity['start_pos']), tuple(identity['end_pos']), solution)
    

def unit_test(size):
     ## Unit test
    maze_bootcamp = Mazebootcamp(size=size, difficulty=1)
    identity = maze_bootcamp.case_generator()
    print(maze_bootcamp.prompt_func(identity))
    solution = solve_maze(identity['grid'], identity['start_pos'], identity['end_pos'])[0]
    fake_output = f"""\n略，
    Final-answer: ```json
    {solution}
    ```"""
    print(fake_output)
    print("Is it correct? ",maze_bootcamp.verify_score(fake_output, identity))
    
if __name__ == '__main__':
    
#     identity = {"grid": [[0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], "start_pos": [0, 0], "end_pos": [19, 19], "difficulty": 3, "seed": None, "solution_path": [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [7, 13], [8, 13], [8, 14], [9, 14], [9, 15], [10, 15], [11, 15], [11, 16], [11, 17], [11, 18], [12, 18], [13, 18], [14, 18], [14, 19], [15, 19], [16, 19], [17, 19], [18, 19], [19, 19]]}
    
#     soulution = """(18,19)是否是P：是的，第18行的列19是P。

# (19,19)是E，正确。

# 因此，这条路径是可行的。

# Final-answer: 
# ```json 
# [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (9, 15), (10, 15), (11, 15), (11, 16), (11, 17), (11, 18), (12, 18), (13, 18), (14, 18), (15, 18), (16, 18), (17, 18), (17, 19), (18, 19), (19, 19)]
# ```"""
    
#     print(Mazebootcamp.verify_score(model_output=soulution, identity=identity))

    unit_test(size=(6,6))