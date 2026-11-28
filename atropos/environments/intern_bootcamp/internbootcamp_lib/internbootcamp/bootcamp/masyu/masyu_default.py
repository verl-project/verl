import ast
import json
import random
import re
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from internbootcamp.bootcamp.base import Basebootcamp
from internbootcamp.libs.masyu.masyu_data_generator import MasyuGenerator
from internbootcamp.libs.masyu.masyu_solver import check_valid_masyu
from internbootcamp.libs.masyu.masyu_z3_solver import solve_masyu_with_z3


class Masyubootcamp(Basebootcamp):
    def __init__(self, size=(6, 6), black_pearls=3, white_pearls=3, seed=None):
        """
        初始化Masyu训练场

        参数:
            size: 谜题大小，(行数, 列数)
            black_pearls: 黑珠数量
            white_pearls: 白珠数量
            seed: 随机种子
        """
        self.size = size
        self.black_pearls = black_pearls
        self.white_pearls = white_pearls
        self.seed = seed
        self.grid = None
        self.solution_path = None
        self.generator = MasyuGenerator()

    def case_generator(self):
        """
        生成一个Masyu谜题

        返回:
            identity: 包含谜题信息的字典
        """
        rows, cols = self.size
        max_attempts = 5  # 最大重试次数

        for attempt in range(max_attempts):
            try:
                # 设置随机种子
                if self.seed is not None:
                    random.seed(self.seed)
                else:
                    random.seed(  time.time())  # 使用当前时间作为种子

                # 生成谜题
                self.grid = self.generator.generate_puzzle(
                    rows,
                    cols,
                    self.black_pearls,
                    self.white_pearls,
                    max_attempts=1000
                )

                # 如果生成失败，尝试减少珠子数量
                if self.grid is None:
                    # print(f"生成谜题失败 (attempt {attempt + 1}/{max_attempts})，尝试减少珠子数量...")
                    self.black_pearls = max(1, self.black_pearls - 1)
                    self.white_pearls = max(1, self.white_pearls - 1)
                    continue

                # 使用Z3求解器获取解决方案
                solution_path = solve_masyu_with_z3(self.grid, timeout=30)

                # 如果无解，重新生成
                if solution_path is None:
                    # print(f"生成的谜题无解 (attempt {attempt + 1}/{max_attempts})，重新生成...")
                    continue

                # 确保解决方案是闭环
                if solution_path and solution_path[0] != solution_path[-1]:
                    solution_path.append(solution_path[0])

                # 创建并返回identity字典
                identity = {
                    'grid': self.grid,
                    'size': self.size,
                    'black_pearls': self.black_pearls,
                    'white_pearls': self.white_pearls,
                    'seed': self.seed,
                    'solution_path': solution_path
                }

                return identity

            except Exception as e:
                # print(f"生成谜题时出错 (attempt {attempt + 1}/{max_attempts}): {str(e)}")
                # 继续下一次尝试
                pass

        # return self._generate_simple_fallback_puzzle()

        return identity

    @staticmethod
    def extract_output(output):
        """
        从解决方案输出中提取路径

        参数:
            output: 模型输出

        返回:
            提取的路径列表
        """
        pattern = r'```json\s*([\s\S]*?)\s*```'
        matches = re.findall(pattern, output)
        if matches:
            python_str = matches[-1]
            try:
                # 尝试解析为Python对象
                result = ast.literal_eval(python_str.strip())
                return result
            except Exception:
                return python_str
        else:
            return None

    @staticmethod
    def print_masyu_str(identity: dict):
        """
        返回Masyu谜题的字符串表示

        参数:
            identity: 谜题信息字典

        返回:
            谜题的字符串表示
        """
        if identity is None:
            return "没有谜题可显示 (identity is None)"

        if 'grid' not in identity or identity['grid'] is None:
            return "没有谜题可显示 (grid is None)"

        result = "Masyu谜题:\n"
        for r in range(len(identity['grid'])):
            line = ""
            for c in range(len(identity['grid'][0])):
                if identity['grid'][r][c] == 'B':
                    line += "● "  # 黑珠
                elif identity['grid'][r][c] == 'W':
                    line += "○ "  # 白珠
                else:
                    line += "· "  # 空格
            result += line + "\n"
        return result

    @classmethod
    def prompt_func(cls, identity: dict) -> str:
        """
        生成提示语

        参数:
            identity: 谜题信息字典

        返回:
            提示语字符串
        """
        if identity is None:
            # print("Warning: identity is None in prompt_func")
            # 创建一个简单的后备谜题
            identity = cls._generate_simple_fallback_puzzle()

        statements = [
            f"""你是一个擅长解决Masyu(珍珠)谜题的智能助手。以下是一个Masyu谜题，请找出满足所有规则的闭环路径。

Masyu规则:
1. 必须画一条封闭的单线环，线不能交叉或分叉。
2. 白珠(○)规则：线必须穿过白珠并在白珠处直行，但必须在白珠的至少一侧转弯。
3. 黑珠(●)规则：线必须在黑珠处转弯，并在黑珠的两侧直行至少一格。
4. 线必须穿过所有珠子，但不必经过所有空格。

谜题如下：

{cls.print_masyu_str(identity)}

请给出形成闭环的完整路径，以坐标序列的形式表示。坐标格式为(行,列)，从0开始计数。
例如：[(0, 0), (0, 1), (1, 1), ..., (0, 0)]

注意：路径必须是一个闭环，所以第一个坐标和最后一个坐标应该相同。
""",
            f"""You are an intelligent assistant specializing in solving Masyu (Pearl) puzzles. Below is a Masyu puzzle that needs to be solved.

Masyu Rules:
1. You must draw a single closed loop that doesn't cross or branch.
2. White pearl (○) rule: The loop must pass straight through white pearls, but must make a turn in at least one of the cells adjacent to the white pearl.
3. Black pearl (●) rule: The loop must make a turn at black pearls, and must go straight through the next cell in both directions.
4. The loop must pass through all pearls, but doesn't need to visit all empty cells.

The puzzle is as follows:

{cls.print_masyu_str(identity)}

Please provide the complete path forming a closed loop as a sequence of coordinates. Coordinates are in (row,column) format, starting from 0.
For example: [(0, 0), (0, 1), (1, 1), ..., (0, 0)]

Note: The path must be a closed loop, so the first and last coordinates should be the same.
"""
        ]
        instruction_following = """\nLet's think step by step and output the final answer with an example markdown formatting: 
Final-answer: ```json
[(0, 0), (0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]
```"""
        return statements[random.randint(0, len(statements) - 1)] + instruction_following

    @classmethod
    def _verify_correction(cls, solution, identity) -> bool:
        """
        验证解决方案是否正确

        参数:
            solution: 用户提供的解决方案
            identity: 谜题信息字典

        返回:
            解决方案是否正确
        """
        # 检查解决方案是否为列表
        if not isinstance(solution, list):
            # print("Error: Solution is not a list")
            return False

        # 检查是否为闭环（首尾相同）
        if not solution or solution[0] != solution[-1]:
            # print("Error: Solution is not a closed loop")
            return False

        # 检查坐标是否有效
        rows, cols = identity['size']
        for r, c in solution:
            if not (0 <= r < rows and 0 <= c < cols):
                # print(f"Error: Invalid coordinates ({r}, {c})")
                return False

        # 检查移动是否有效（只能上下左右移动一格）
        for i in range(len(solution) - 1):
            r1, c1 = solution[i]
            r2, c2 = solution[i + 1]
            if not ((abs(r1 - r2) == 1 and c1 == c2) or (r1 == r2 and abs(c1 - c2) == 1)):
                # print(f"Error: Invalid move from ({r1}, {c1}) to ({r2}, {c2})")
                return False

        # 检查是否经过所有珠子
        pearl_positions = []
        for r in range(rows):
            for c in range(cols):
                if identity['grid'][r][c] in ['B', 'W']:
                    pearl_positions.append((r, c))

        for pos in pearl_positions:
            if pos not in solution:
                # print(f"Error: Path does not pass through pearl at {pos}")
                return False

        # 检查白珠规则
        for r in range(rows):
            for c in range(cols):
                if identity['grid'][r][c] == 'W':
                    if not cls._check_white_pearl_rule(solution, r, c):
                        # print(f"Error: White pearl rule violated at ({r}, {c})")
                        return False

        # 检查黑珠规则
        for r in range(rows):
            for c in range(cols):
                if identity['grid'][r][c] == 'B':
                    if not cls._check_black_pearl_rule(solution, r, c):
                        # print(f"Error: Black pearl rule violated at ({r}, {c})")
                        return False

        return True

    @staticmethod
    def _check_white_pearl_rule(path, r, c):
        """
        检查白珠规则：线必须穿过白珠并在白珠处直行，但必须在白珠的至少一侧转弯

        参数:
            path: 路径
            r, c: 白珠坐标

        返回:
            是否满足白珠规则
        """
        # 找到白珠在路径中的索引
        try:
            idx = path.index((r, c))
        except ValueError:
            return False

        # 获取前后的点
        prev_idx = (idx - 1) % (len(path) - 1)
        next_idx = (idx + 1) % (len(path) - 1)

        prev_r, prev_c = path[prev_idx]
        next_r, next_c = path[next_idx]

        # 检查是否直行
        is_straight = (prev_r == next_r) or (prev_c == next_c)
        if not is_straight:
            return False

        # 检查至少一侧是否转弯
        # 获取前前和后后的点
        prev_prev_idx = (prev_idx - 1) % (len(path) - 1)
        next_next_idx = (next_idx + 1) % (len(path) - 1)

        prev_prev_r, prev_prev_c = path[prev_prev_idx]
        next_next_r, next_next_c = path[next_next_idx]

        # 检查前一侧是否转弯
        prev_turn = not ((prev_prev_r == prev_r == r) or (prev_prev_c == prev_c == c))

        # 检查后一侧是否转弯
        next_turn = not ((next_r == next_next_r == r) or (next_c == next_next_c == c))

        return prev_turn or next_turn

    @staticmethod
    def _check_black_pearl_rule(path, r, c):
        """
        检查黑珠规则：线必须在黑珠处转弯，并在黑珠的两侧直行至少一格

        参数:
            path: 路径
            r, c: 黑珠坐标

        返回:
            是否满足黑珠规则
        """
        # 找到黑珠在路径中的索引
        try:
            idx = path.index((r, c))
        except ValueError:
            return False

        # 获取前后的点
        prev_idx = (idx - 1) % (len(path) - 1)
        next_idx = (idx + 1) % (len(path) - 1)

        prev_r, prev_c = path[prev_idx]
        next_r, next_c = path[next_idx]

        # 检查是否转弯
        is_turn = not ((prev_r == next_r) or (prev_c == next_c))
        if not is_turn:
            return False

        # 检查两侧是否直行至少一格
        # 获取前前和后后的点
        prev_prev_idx = (prev_idx - 1) % (len(path) - 1)
        next_next_idx = (next_idx + 1) % (len(path) - 1)

        prev_prev_r, prev_prev_c = path[prev_prev_idx]
        next_next_r, next_next_c = path[next_next_idx]

        # 检查前一侧是否直行
        prev_straight = (prev_prev_r == prev_r) or (prev_prev_c == prev_c)

        # 检查后一侧是否直行
        next_straight = (next_r == next_next_r) or (next_c == next_next_c)

        return prev_straight and next_straight


if __name__ == '__main__':
    # 单元测试
    try:
        masyu_bootcamp = Masyubootcamp(size=(6, 6), black_pearls=3, white_pearls=3)
        identity = masyu_bootcamp.case_generator()

        if identity is None:
            print("Error: Failed to generate puzzle")
            exit(1)

        print(masyu_bootcamp.prompt_func(identity))

        # 使用正确的解决方案进行测试
        solution = identity['solution_path']

        fake_output = f"""\n略，
        Final-answer: ```json
        {solution}
        ```"""
        print(fake_output)
        print("Is it correct? ", masyu_bootcamp.verify_score(fake_output, identity))

    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback

        traceback.print_exc()

