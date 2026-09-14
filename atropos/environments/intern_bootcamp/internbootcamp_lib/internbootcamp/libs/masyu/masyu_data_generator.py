import os
import random
import time
import json
from .masyu_solver import check_valid_masyu


class MasyuGenerator:
    def __init__(self):
        self.npuzzles = 0

    def generate_puzzle(self, rows, cols, black_pearls, white_pearls, max_attempts=10000):
        """
        生成一个 Masyu 谜题，简化了生成和验证过程
        """
        # print(f"Generating {rows}x{cols} puzzle with {black_pearls} black and {white_pearls} white pearls...")

        for attempt in range(max_attempts):
            if attempt % 100 == 0:
                # print(f"Attempt {attempt}/{max_attempts}...")
                pass

            # 创建空网格
            grid = [[' ' for _ in range(cols)] for _ in range(rows)]

            # 随机放置珠子
            if not self._place_random_pearls(grid, rows, cols, black_pearls, white_pearls):
                continue

            # 检查谜题是否有唯一解
            if check_valid_masyu(grid):
                # print(f"Found valid puzzle after {attempt + 1} attempts")
                return grid

        # print(f"Failed to generate puzzle after {max_attempts} attempts")
        return None

    def _place_random_pearls(self, grid, rows, cols, black_pearls, white_pearls):
        """随机放置珠子，考虑珠子之间的规则性"""
        total_pearls = black_pearls + white_pearls

        # 检查是否有足够的单元格
        if total_pearls > rows * cols:
            return False

        # 获取所有可用单元格
        available_cells = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(available_cells)

        # 放置黑珠和白珠时，确保不会互相冲突
        for i in range(black_pearls):
            if i < len(available_cells):
                r, c = available_cells[i]
                grid[r][c] = 'B'  # 放置黑珠

        for i in range(black_pearls, black_pearls + white_pearls):
            if i < len(available_cells):
                r, c = available_cells[i]
                grid[r][c] = 'W'  # 放置白珠

        # 只进行最基本的验证，放宽规则
        return True

    def print_puzzle(self, grid):
        """打印谜题"""
        if grid is None:
            # print("No puzzle to display")
            return

        nrows = len(grid)
        ncols = len(grid[0]) if nrows > 0 else 0

        # print("Puzzle:")
        for r in range(nrows):
            line = ""
            for c in range(ncols):
                if grid[r][c] == 'B':
                    line += "● "
                elif grid[r][c] == 'W':
                    line += "○ "
                else:
                    line += "  "
            # print(line)

    def batch_generate_puzzles(self, count, min_rows, max_rows, min_cols, max_cols, min_black_pearls, max_black_pearls,
                               min_white_pearls, max_white_pearls, output_dir="generated_puzzles", max_attempts=1000):
        """
        批量生成 Masyu 谜题，并保存为 JSON 文件
        增加了随机生成参数的功能
        """
        os.makedirs(output_dir, exist_ok=True)

        valid_puzzles = []  # 用于存储生成的有效谜题

        for i in range(count):
            # print(f"\nGenerating puzzle {i + 1}/{count}...")

            # 随机设置谜题的行数、列数、黑珠和白珠数目，减少珠子数量，避免过多的约束
            rows = random.randint(min_rows, max_rows)
            cols = random.randint(min_cols, max_cols)
            black_pearls = random.randint(min_black_pearls, max_black_pearls)
            white_pearls = random.randint(min_white_pearls, max_white_pearls)

            # 尝试生成一个有效的谜题
            puzzle = self.generate_puzzle(rows, cols, black_pearls, white_pearls, max_attempts)
            if puzzle:
                puzzle_data = {
                    "id": f"masyu_puzzle_{i + 1}",
                    "rows": rows,
                    "cols": cols,
                    "puzzle": puzzle
                }
                valid_puzzles.append(puzzle_data)

                # 保存单个谜题到 JSON 文件
                puzzle_file = os.path.join(output_dir, f"masyu_puzzle_{i + 1}.json")
                with open(puzzle_file, "w") as f:
                    json.dump(puzzle_data, f, indent=2)
                # print(f"Saved puzzle {i + 1} to {puzzle_file}")
            else:
                # print(f"Failed to generate puzzle {i + 1}")
                pass

        # 将所有有效谜题保存到一个大文件中
        dataset_file = os.path.join(output_dir, "masyu_dataset.json")
        with open(dataset_file, "w") as f:
            json.dump(valid_puzzles, f, indent=2)
        # print(f"Saved all generated puzzles to {dataset_file}")


def main():
    # 设置生成谜题的参数
    generator = MasyuGenerator()

    # 设置随机范围，生成 1000 个谜题
    generator.batch_generate_puzzles(
        count=50,  # 生成谜题个数
        min_rows=4, max_rows=6,  # 随机生成 4x4 到 6x6 大小的谜题
        min_cols=4, max_cols=6,  # 随机生成 4x4 到 6x6 大小的谜题
        min_black_pearls=2, max_black_pearls=3,  # 随机生成 2 到 3 个黑珠
        min_white_pearls=2, max_white_pearls=3,  # 随机生成 2 到 3 个白珠
        output_dir="generated_puzzles",  # 保存到当前目录下的 generated_puzzles 文件夹
        max_attempts=1000  # 每个谜题的最大尝试次数
    )

if __name__ == "__main__":
    main()