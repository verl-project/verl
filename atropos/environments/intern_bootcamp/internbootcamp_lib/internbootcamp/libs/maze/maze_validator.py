def validate_maze_solution(maze, start_pos, end_pos, solution):
    """
    验证迷宫解决方案是否有效

    参数:
        maze: 迷宫的二维数组（0 表示通路，1 表示墙）
        start_pos: 起点位置
        end_pos: 终点位置
        solution: 解决方案路径

    返回:
        is_valid: 解决方案是否有效
    """
    # 检查解决方案是否为空
    if not solution:
        return False

    # 检查起点和终点
    if solution[0] != start_pos or solution[-1] != end_pos:
        return False

    height = len(maze)
    width = len(maze[0]) if height > 0 else 0

    # 检查每一步是否有效
    for i, (x, y) in enumerate(solution):
        # print(f"Checking position {x}, {y}...")
        if not (0 <= x < height and 0 <= y < width):
            # print(f"Position {x}, {y} is out of bounds.")
            return False
        if maze[x][y] == 1:
            # print(f"Position {x}, {y} is a wall.")
            return False
        if i > 0:
            prev_x, prev_y = solution[i - 1]
            if not ((abs(x - prev_x) == 1 and y == prev_y) or (abs(y - prev_y) == 1 and x == prev_x)):
                # print(f"Position {x}, {y} is not adjacent to {prev_x}, {prev_y}.")
                return False

    return True


def validate_and_solve_mazes(mazes):
    """
    批量验证和解答迷宫

    参数:
        mazes: 迷宫列表

    返回:
        results: 验证和解答结果列表
    """
    from maze_solver import solve_maze, is_path_exist

    results = []
    for maze_data in mazes:
        maze_id = maze_data["id"]
        maze_grid = maze_data["grid"]

        # 获取起点和终点（如果有指定）
        start_pos = maze_data.get("start_pos", (0, 0))
        end_pos = maze_data.get("end_pos", (len(maze_grid) - 1, len(maze_grid[0]) - 1))

        # 验证迷宫是否可解
        has_path = is_path_exist(maze_grid, start_pos, end_pos)

        # 如果可解，尝试找到路径
        solution = None
        if has_path:
            solution, _ = solve_maze(maze_grid, start_pos, end_pos)

        results.append({
            "id": maze_id,
            "has_path": has_path,
            "solution": solution
        })

    return results


if __name__ == "__main__":
    # 测试验证器
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    start_pos = (0, 0)
    end_pos = (4, 4)

    # 有效路径
    valid_solution = [
        (0, 0), (1, 0), (2, 0), (2, 1), (2, 2),
        (3, 2), (4, 2), (4, 3), (4, 4)
    ]

    # 无效路径（不连续）
    invalid_solution = [
        (0, 0), (1, 0), (2, 0), (2, 2),
        (3, 2), (4, 2), (4, 3), (4, 4)
    ]

    print("有效路径验证结果:", validate_maze_solution(maze, start_pos, end_pos, valid_solution))
    print("无效路径验证结果:", validate_maze_solution(maze, start_pos, end_pos, invalid_solution))
