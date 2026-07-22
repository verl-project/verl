import random
import sys
import os

# 获取 generator.py 所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))

# 将父目录加入 sys.path
sys.path.append(parent_dir)

from libs.Skyscrapers.main import solver_main

def generate_puzzle(n:int):
    """
    Generate an n x n Skyscrapers puzzle with valid external hints.

    Args:
        n (int): The size of the grid.

    Returns:
        tuple: A tuple containing the generated grid and the hints (top, bottom, left, right).
    """
    # Generate a valid n x n grid where each row and column contains numbers 1 to n without repetition
    def generate_valid_grid(n):
        while True:
            grid = [random.sample(range(1, n + 1), n) for _ in range(n)]
            transposed = list(zip(*grid))
            if all(len(set(column)) == n for column in transposed):
                return grid

    grid = generate_valid_grid(n)

    # Calculate external hints
    def calculate_hints_top(grid):
        n = len(grid)
        hints = []
        for col in range(n):
            max_height = 0
            visible_count = 0
            for row in range(n):
                if grid[row][col] > max_height:
                    max_height = grid[row][col]
                    visible_count += 1
            hints.append(visible_count)
        return hints

    def calculate_hints_bottom(grid):
        n = len(grid)
        hints = []
        for col in range(n):
            max_height = 0
            visible_count = 0
            for row in range(n - 1, -1, -1):
                if grid[row][col] > max_height:
                    max_height = grid[row][col]
                    visible_count += 1
            hints.append(visible_count)
        return hints

    def calculate_hints_left(grid):
        hints = []
        for row in grid:
            max_height = 0
            visible_count = 0
            for height in row:
                if height > max_height:
                    max_height = height
                    visible_count += 1
            hints.append(visible_count)
        return hints

    def calculate_hints_right(grid):
        hints = []
        for row in grid:
            max_height = 0
            visible_count = 0
            for height in reversed(row):
                if height > max_height:
                    max_height = height
                    visible_count += 1
            hints.append(visible_count)
        return hints

    top_hints = calculate_hints_top(grid)
    bottom_hints = calculate_hints_bottom(grid)
    left_hints = calculate_hints_left(grid)
    right_hints = calculate_hints_right(grid)

    return grid, top_hints, bottom_hints, left_hints, right_hints

if __name__ == "__main__":
    n = 5  # Example size
    grid, top_hints, bottom_hints, left_hints, right_hints = generate_puzzle(n)

    # print("Generated Grid:")
    # for row in grid:
    #     print(row)

    # print("\nHints:")
    # print("Top:", top_hints)
    # print("Bottom:", bottom_hints)
    # print("Left:", left_hints)
    # print("Right:", right_hints)

    solver_main(n,left_hints,right_hints,top_hints,bottom_hints)