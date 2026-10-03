import os
import argparse
import glob
from ..slitherlink.slitherlink_generator import SlitherlinkSolver  # 引入求解器


# 验证谜题的函数
def verify_puzzle(input_file, output_file=None, check_only=False):
    # 读取谜题文件
    print(f"验证谜题: {input_file}")
    puzzle = []
    with open(input_file, 'r') as f:
        for line in f:
            row = [None if cell == '.' else int(cell) for cell in line.strip()]
            puzzle.append(row)

    # 创建 SlitherlinkSolver 实例并加载谜题
    solver = SlitherlinkSolver()
    solver.cells = puzzle
    solver.height = len(puzzle)
    solver.width = len(puzzle[0])

    if check_only:
        # 仅验证谜题是否有解
        if solver.solve():
            print(f"谜题 {input_file} 有解！")
        else:
            print(f"谜题 {input_file} 无解！")
    else:
        # 求解谜题并保存结果
        if solver.solve():
            solution = solver.solution
            print(f"谜题 {input_file} 的解答已找到并保存到 {output_file}")
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(', '.join(map(str, solution)))
        else:
            print(f"谜题 {input_file} 无解！")


# 处理谜题文件的函数
def process_puzzles(puzzle_folder, solution_folder, check_only=False):
    # 获取所有谜题文件
    puzzle_files = glob.glob(os.path.join(puzzle_folder, '*.txt'))  # 假设谜题文件为 txt 格式
    solution_files = glob.glob(os.path.join(solution_folder, '*.txt'))  # 假设解决方案为 txt 格式

    # 验证每个谜题文件
    for puzzle_file in puzzle_files:
        # 创建对应的输出文件路径
        puzzle_filename = os.path.basename(puzzle_file)
        solution_filename = os.path.splitext(puzzle_filename)[0] + '_solution.txt'
        output_file = os.path.join(solution_folder, solution_filename)

        # 调用验证函数
        verify_puzzle(puzzle_file, output_file if not check_only else None, check_only)


# 主函数
def main():
    # 指定谜题和解决方案文件夹路径
    puzzle_folder = './puzzles'  # 修改为你谜题文件所在的文件夹路径
    solution_folder = './solutions'  # 修改为你解决方案文件所在的文件夹路径

    # 设置是否仅验证有效性
    check_only = False  # 如果你只想验证谜题有效性，可以设为 True

    # 处理所有谜题
    process_puzzles(puzzle_folder, solution_folder, check_only)


if __name__ == "__main__":
    main()
