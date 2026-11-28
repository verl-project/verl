import ast
import json
import random
import re
from typing import Dict, List, Any, Tuple, Optional

from internbootcamp.bootcamp.base import Basebootcamp
from internbootcamp.libs.slitherlink.slitherlink_generator import SlitherlinkSolver, generate_puzzle
from internbootcamp.libs.slitherlink.slitherlink_verifier import verify_puzzle


class Slitherlinkbootcamp(Basebootcamp):
    """
    Slitherlink谜题解决系统，用于生成谜题、提供提示和验证解答
    """

    def __init__(self, size: Tuple[int, int] = (5, 5), difficulty: str = 'medium', seed: Optional[int] = None):
        """
        初始化Slitherlink解答系统

        Args:
            size: 谜题大小，(行数,列数)
            difficulty: 难度级别 ('easy','medium','hard')
            seed: 随机种子，用于生成可重复的谜题
        """
        self.size = tuple(size)
        self.difficulty = difficulty
        self.grid = None
        self.solution = None
        self.seed = seed

        if seed is not None:
            random.seed(seed)

    def case_generator(self) -> Dict[str, Any]:
        """
        生成一个Slitherlink谜题

        返回:
            包含谜题信息的字典
        """
        rows, cols = self.size

        # 调用现有的generate_puzzle函数生成谜题
        puzzle = generate_puzzle(rows, cols, self.difficulty)

        # 确保谜题有解
        solver = SlitherlinkSolver()
        solver.cells = puzzle
        solver.height = rows
        solver.width = cols

        if solver.solve():
            identity = {
                'grid': puzzle,
                'size': self.size,
                'difficulty': self.difficulty,
                'solution': solver.solution,
                'seed': self.seed
            }
            return identity
        else:
            # 如果无解则重新生成
            return self.case_generator()

    @staticmethod
    def extract_output(output: str) -> List[int]:
        """
        从模型输出中提取解答

        Args:
            output: 模型输出的文本

        Returns:
            边的列表
        """
        pattern = r'```json\s*([\s\S]*?)\s*```'
        matches = re.findall(pattern, output)
        if matches:
            python_str = matches[-1]
            try:
                # 尝试解析为Python对象
                result = ast.literal_eval(python_str.strip())
                if isinstance(result, list):
                    return result
                return []
            except Exception:
                # 如果解析失败，尝试查找数字列表
                number_pattern = r'\[([0-9, ]+)\]'
                number_matches = re.findall(number_pattern, python_str)
                if number_matches:
                    try:
                        return [int(x.strip()) for x in number_matches[0].split(',')]
                    except:
                        return []
        return []

    @staticmethod
    def print_puzzle_str(identity: Dict[str, Any]) -> str:
        """
        返回谜题的字符串表示

        Args:
            identity: 包含谜题信息的字典

        Returns:
            谜题的字符串表示
        """
        if identity.get('grid') is None:
            return "没有谜题可显示"

        result = "Slitherlink谜题:\n"
        for row in identity['grid']:
            line = ""
            for cell in row:
                line += str(cell) if cell is not None else "."
            result += line + "\n"
        return result

    @staticmethod
    def visualize_solution(identity: Dict[str, Any], solution: List[int]) -> str:
        """
        可视化解决方案

        Args:
            identity: 包含谜题信息的字典
            solution: 解决方案（边的列表）

        Returns:
            解决方案的可视化字符串
        """
        rows, cols = identity['size']
        grid = identity['grid']

        # 创建点的网格 (rows+1) x (cols+1)
        points = [['.' for _ in range(cols + 1)] for _ in range(rows + 1)]

        # 创建水平边和垂直边的网格
        h_edges = [[' ' for _ in range(cols)] for _ in range(rows + 1)]
        v_edges = [[' ' for _ in range(cols + 1)] for _ in range(rows)]

        # 填充解决方案中的边
        for edge in solution:
            edge = edge - 1  # 调整索引（假设边从1开始编号）
            vert_edges = rows * (cols + 1)

            if edge < vert_edges:
                # 垂直边
                edge_row = edge // (cols + 1)
                edge_col = edge % (cols + 1)
                v_edges[edge_row][edge_col] = '|'
            else:
                # 水平边
                edge -= vert_edges
                edge_row = edge // cols
                edge_col = edge % cols
                h_edges[edge_row][edge_col] = '-'

        # 生成可视化输出
        result = "解决方案:\n"
        for i in range(rows + 1):
            # 打印水平边
            h_line = ""
            for j in range(cols + 1):
                h_line += points[i][j]
                if j < cols:
                    h_line += h_edges[i][j]
            result += h_line + "\n"

            # 打印垂直边和数字
            if i < rows:
                v_line = ""
                for j in range(cols + 1):
                    v_line += v_edges[i][j]
                    if j < cols:
                        cell_value = grid[i][j]
                        v_line += str(cell_value) if cell_value is not None else " "
                result += v_line + "\n"

        return result

    @classmethod
    def prompt_func(cls, identity: Dict[str, Any]) -> str:
        """
        生成提示信息

        Args:
            identity: 包含谜题信息的字典

        Returns:
            提示信息字符串
        """
        statements = [
            f"""你是一个擅长解决Slitherlink谜题的智能助手。以下是一个Slitherlink谜题，请找出解决方案。

Slitherlink规则：
1. 在点之间连线形成一个单一的闭环
2. 线不能交叉或分叉
3. 每个数字表示该格子周围经过的线段数量
4. 空格子（标记为.）可以经过任意数量的线段（0-4）

谜题如下：

{cls.print_puzzle_str(identity)}

请给出解决方案，以边的列表形式表示。边的编号从1开始，先从左到右、从上到下编号所有垂直边，然后从左到右、从上到下编号所有水平边。

请确保你的解决方案形成一个单一的闭环，不包含任何交叉或分叉。
""",
            f"""You are an intelligent assistant specializing in solving Slitherlink puzzles. Below is a Slitherlink puzzle that needs to be solved.

Slitherlink Rules:
1. Connect dots with lines to form a single closed loop
2. Lines cannot cross or branch
3. Numbers indicate how many lines surround that cell
4. Empty cells (marked with .) can be surrounded by any number of lines (0-4)

The puzzle is as follows:

{cls.print_puzzle_str(identity)}

Please provide the solution as a list of edges. Edges are numbered starting from 1, first numbering all vertical edges from left to right, top to bottom, then all horizontal edges from left to right, top to bottom.

Make sure your solution forms a single closed loop without any crossings or branches.
"""
        ]
        instruction_following = """\nLet's think step by step and output the final answer with an example markdown formatting: 
Final-answer: ```json
[edge1, edge2, edge3, edge4, edge5, ... , edgeN]
 ]
```"""
        return statements[random.randint(0, len(statements) - 1)] + instruction_following

    @classmethod
    def _verify_correction(cls, solution: List[int], identity: Dict[str, Any]) -> bool:
        """
        验证解决方案是否正确

        Args:
            solution: 解决方案（边的列表）
            identity: 包含谜题信息的字典

        Returns:
            解决方案是否正确
        """
        if not solution:
            return False

        solver = SlitherlinkSolver()
        solver.cells = identity['grid']
        solver.height = identity['size'][0]
        solver.width = identity['size'][1]

        # 验证单元格约束
        for row in range(solver.height):
            for col in range(solver.width):
                cell_value = solver.cells[row][col]
                if cell_value is not None:
                    # 获取单元格周围的边
                    cell_id = row * solver.width + col
                    edges = solver.get_cell_edges(cell_id)

                    # 计算解决方案中包含的边数
                    edge_count = sum(1 for edge in edges if edge + 1 in solution)

                    # 验证边数是否与单元格值匹配
                    if edge_count != cell_value:
                        return False

        # 验证是否形成有效的闭环
        return solver.validate(solution)

if __name__ == '__main__':
    # 单元测试
    slitherlink_bootcamp = Slitherlinkbootcamp(size=(5, 5), difficulty='medium')
    identity = slitherlink_bootcamp.case_generator()
    print(slitherlink_bootcamp.prompt_func(identity))

    # 使用正确的解决方案进行测试
    solution = identity['solution']
    fake_output = f"""\n经过分析，
    Final-answer: ```json
    {solution}
    ```"""
    print(fake_output)
    

    # 可视化解决方案
    print(slitherlink_bootcamp.visualize_solution(identity, solution))
    print("Is it correct? ", slitherlink_bootcamp.verify_score(fake_output, identity))
