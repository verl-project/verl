import re
import json
import random

from internbootcamp.bootcamp.base import Basebootcamp
from internbootcamp.libs.arrowmaze.maze_generator import generate_arrow_maze
class Arrowmazebootcamp(Basebootcamp):
    def __init__(self, size:tuple = (6,6), start_pos:tuple = (0,0), end_pos:tuple = (5,5), max_solution_step:int = 5, seed:int = None):
        self.size = size
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.max_solution_step = max_solution_step
        self.seed = seed
    def case_generator(self):
        if self.max_solution_step < 2:
            raise ValueError
        grid = generate_arrow_maze(self.size[0], self.size[1], tuple(self.start_pos), tuple(self.end_pos), max_attempts= 20 ,max_solution_step=self.max_solution_step, seed=self.seed)
        return {
            "input_grid": grid,
            "start_position": self.start_pos
        }
    
    def prompt_func(self, identity) -> str:
        """
        Process the input_data and return the processed prompt.
        
        Args:
            question_ori: The question to be processed.
        
        Returns:
            str: The processed prompt.
        """

        statements = [f"""You are an intelligent assistant specializing in solving custom puzzle problems. Below is a specific rule defined for a custom puzzle. Your task is to apply this rule accurately to the provided question.

    ### Instructions:

    1. Thoroughly understand the rule provided. If needed, break down the rule into simpler components or steps.
    2. Apply the rule carefully to address the question presented.
    3. Verify your answer to ensure it aligns with the rule and the context of the puzzle.
### Puzzle Rule:
1.The maze consists of a grid with an arrow in each grid cell pointing in one of eight directions up, down, left, right, or diagonally.
2.The maze has a well-defined start and end point.
3.The player starts at the starting point, moves to the next grid cell in the direction indicated by the arrow, and then continues to move as indicated by the arrow in the new grid.
4.The player must move strictly in the direction indicated by the arrows and cannot go in the opposite direction or choose another path.
5.The game is won when the player successfully reaches the end from the starting point.

### Question

Now the grid of the arrow maze is:\n{identity["input_grid"]}
The start position is {identity["start_position"]} and the end position is 'o' in the grid.

The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
""",
f"""
The arrow maze is a puzzle game. The rule of arrow maze is:
1.The maze consists of a grid with an arrow in each grid cell pointing in one of eight directions up, down, left, right, or diagonally.
2.The maze has a well-defined start and end point.
3.The player starts at the starting point, moves to the next grid cell in the direction indicated by the arrow, and then continues to move as indicated by the arrow in the new grid.
4.The player must move strictly in the direction indicated by the arrows and cannot go in the opposite direction or choose another path.
5.The game is won when the player successfully reaches the end from the starting point.

Now the grid of the arrow maze is:\n{identity["input_grid"]}
The start position is {identity["start_position"]} and the end position is 'o' in the grid.
"""
,
f"""你是一个擅长解决自定义谜题问题的智能助手。以下是为一个自定义谜题所定义的特定规则。你的任务是将该规则准确应用到所提供的问题上。
说明：
透彻理解所提供的规则。如有需要，将规则拆解为更简单的组成部分或步骤。
仔细运用规则来解决给出的问题。
核实你的答案，确保其与规则以及谜题的情境相符。
谜题规则：
迷宫由一个网格构成，每个网格单元格中有一个箭头，指向八个方向之一，即上、下、左、右或对角线方向。
迷宫有明确规定的起点和终点。
玩家从起点出发，按照箭头所指方向移动到下一个网格单元格，然后继续按照新网格中箭头所指方向移动。
玩家必须严格按照箭头指示的方向移动，不能朝相反方向移动或选择其他路径。
当玩家成功从起点抵达终点时，游戏获胜。
问题
现在箭头迷宫的网格如下：
\n{identity["input_grid"]}

起点位置是 (0, 0)，终点位置是网格中的 “○”。
答案需要按顺序指出每个转折点的位置，0 表示不在路径上的点。
"""

]

        instr = statements[random.randint(0,len(statements)-1)]
        instruction_following = """Let's think step by step and output the final answer with an example json formatting: 
Final-answer: ```json
[[A, B, C, D],
[E, F, G, H],
[I, J, K, L],
[M, N, O, P]]
```"""
        prompt = instr + '\n' + instruction_following
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
        pattern = pattern = r'```json\s*([\s\S]*?)\s*```'
        matches = re.findall(pattern, output)
        # Final-answer:{'A': [(3, 1)],
        # 'B': [(1, 4)],
        # 'C': [(2, 3)],
        # 'D': [(5, 2)],
        # 'E': [(4, 5)]}
        if matches:
            # 获取 JSON 字符串
            json_str = matches[-1]
            # print('match?', json_str)
            # print('solution generated? first lines', output[:200])
            # print('solution generated? last lines', output[-200:])
            # 替换单引号为双引号，将元组表示改为列表表示
            json_str = json_str.replace("'", '"').replace("(", "[").replace(")", "]")
            try:
                # 解析 JSON 字符串为 Python 字典
                result_dict = json.loads(json_str)
                return result_dict
            except json.JSONDecodeError as e:
                return json_str
        else:
            return None
        
    @staticmethod 
    def _verify_correction(solution,identity:dict)->bool:
        """
        Validate whether a candidate_path in puzzle's format (e.g. "[[1 0 0,0 0 0,0 0 2]]")
        is a correct solution to the arrow maze.

        Parameters
        ----------
        grid : list[list[str]]
            A 2D grid of arrow symbols or '○'.
            Example:
            [
            ['→', '↙', '↓'],
            ['↖', '↓', '↙'],
            ['↑', '←', '○'],
            ]
        start_position : (int, int)
            (row, col) of the starting cell.
        answer : list
            The proposed solution in the format "[[...]]"
            0 => not on path
            1 => first visited cell
            2 => second visited cell
            etc.

        Returns
        -------
        bool
            True if the path is valid, False otherwise.
        """
        if not 'input_grid' in identity:
            raise ValueError("input_grid is not in identity")
        else:
            input_grid = identity['input_grid']
        
        if not 'start_position' in identity:
            start_position = (0, 0)
        else:
            start_position = tuple(identity['start_position'])
        input_grid = json.loads(input_grid) if type(input_grid) == str else input_grid
        candidate_grid = json.loads(solution) if type(solution) == str else solution
        # Directions dictionary: maps arrow symbol -> (dr, dc)
        DIRECTIONS = {
            '↑':  (-1,  0),
            '↓':  ( 1,  0),
            '←':  ( 0, -1),
            '→':  ( 0,  1),
            '↖':  (-1, -1),
            '↗':  (-1,  1),
            '↙':  ( 1, -1),
            '↘':  ( 1,  1),
        }

        rows = len(input_grid)
        cols = len(input_grid[0]) if rows > 0 else 0

        def in_bounds(r, c):
            return 0 <= r < rows and 0 <= c < cols

        
        
        # candidate_grid = answer

        # Sanity check: the candidate_grid should match the same dimensions as 'grid'
        if len(candidate_grid) != rows:
            return False
        for row_vals in candidate_grid:
            if len(row_vals) != cols:
                return False
        
        # 2. Extract the labeled cells: (label, (row, col))
        #    We only care about label > 0
        labeled_cells = []
        for r in range(rows):
            for c in range(cols):
                label = candidate_grid[r][c]
                if label > 0:
                    labeled_cells.append((label, (r, c)))
        
        # If no labeled cells, invalid
        if not labeled_cells:
            return False
        
        # 3. Sort by label ascending
        labeled_cells.sort(key=lambda x: x[0])  # sort by label number
        # This gives us an ordered path: [ (1, (r1,c1)), (2, (r2,c2)), ... ]

        # 4. The path in terms of coordinates:
        path = [cell_coord for _, cell_coord in labeled_cells]

        # 5. Check that label "1" is at start_position
        if path[0] != start_position:
            return False

        # 6. Validate each consecutive step in path
        for i in range(len(path) - 1):
            (r1, c1) = path[i]
            (r2, c2) = path[i + 1]

            if not in_bounds(r1, c1) or not in_bounds(r2, c2):
                return False

            # If the current cell is the end symbol '○' but we still have more steps, invalid
            if input_grid[r1][c1] == '○':
                return False

            # Arrow in the current cell:
            arrow_symbol = input_grid[r1][c1]
            if arrow_symbol not in DIRECTIONS:
                return False  # not an arrow and not the end symbol

            (dr, dc) = DIRECTIONS[arrow_symbol]
            delta_r = r2 - r1
            delta_c = c2 - c1

            # Must move in a positive integer multiple of (dr, dc).
            if dr == 0 and dc == 0:
                return False  # shouldn't happen with valid arrows

            # Horizontal or vertical
            if dr == 0:
                # vertical movement is zero => must move horizontally
                # check we didn't move in row, must move in col
                if delta_r != 0:
                    return False
                # direction must match sign of dc
                if dc > 0 and delta_c <= 0:
                    return False
                if dc < 0 and delta_c >= 0:
                    return False
            elif dc == 0:
                # horizontal movement is zero => must move in row
                if delta_c != 0:
                    return False
                if dr > 0 and delta_r <= 0:
                    return False
                if dr < 0 and delta_r >= 0:
                    return False
            else:
                # diagonal
                if delta_r == 0 or delta_c == 0:
                    return False  # can't be diagonal if one is zero
                if (dr > 0 and delta_r <= 0) or (dr < 0 and delta_r >= 0):
                    return False
                if (dc > 0 and delta_c <= 0) or (dc < 0 and delta_c >= 0):
                    return False
                # check integer multiples
                if (delta_r % dr) != 0 or (delta_c % dc) != 0:
                    return False
                factor_r = delta_r // dr
                factor_c = delta_c // dc
                if factor_r != factor_c or factor_r <= 0:
                    return False

        # 7. Check last labeled cell is the '○' cell
        last_r, last_c = path[-1]
        if not in_bounds(last_r, last_c):
            return False
        if input_grid[last_r][last_c] != '○':
            return False

        # If all checks pass, it's a valid solution
        return True
            

if __name__ == "__main__":
    # A small 3×3 arrow maze with start=(0, 0):
    # Grid (3x3):
    #   →   ↙   ↓
    #   ↖   ↓   ↙
    #   ↑   ←   ○
    grid = [
        ['→', '↓', '↓'],
        ['↖', '↓', '↙'],
        ['↑', '→', '○'],
    ]
    start_position = (0, 0)

    # Example candidate_path string:
    # "[[1 0 0,0 0 0,0 0 2]]"
    # This claims:
    #   row=0 col=0 => 1 (start)
    #   row=2 col=2 => 2 (end)
    candidate_path_str = """现在，我需要构建一个3x3的数组，每个位置标注它是第几个被访问的点。起点(0,0)是1，(1,1)是2，终点(0,2)是3。其他位置都是0。

所以，数组应该是：

[
 [1, 0, 3],
 [0, 2, 0],
 [0, 0, 0]
]

但是，问题中的示例看起来像是一个更大的数组，可能包含更多的行和列。所以，我需要仔细检查是否正确。或者，是否需要包括所有可能的移动点，比如从(1,1)到(0,2)是否经过其他点？其实，在这一步，移动是直接对角线，所以只经过(1,1)和(0,2)两点。

因此，最终的数组应该显示路径上的点，并按顺序编号。
</think>

Final-answer: ```json
[
  [1, 0, 0],
  [0, 0, 0],
  [0, 0, 2]
]
```
    """
    is_valid = Arrowmazebootcamp.verify_score(model_output=candidate_path_str,
                             identity={'input_grid': [['↘', '↙', '↖'], 
                                                      ['↗', '↗', '↗'], 
                                                      ['→', '↘', '○']], 'start_position': [0, 0]})
    
    print("Is the candidate path valid?", is_valid)
