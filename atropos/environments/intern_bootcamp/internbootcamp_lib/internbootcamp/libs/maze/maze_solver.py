from collections import deque


def solve_maze(maze, start_pos=(0, 0), end_pos=None):
    """
    使用广度优先搜索(BFS)求解迷宫

    参数:
        maze: 迷宫的二维数组（0 表示通路，1 表示墙）
        start_pos: 起点位置，默认为(0, 0)
        end_pos: 终点位置，默认为右下角

    返回:
        path: 从起点到终点的路径列表，如果没有路径则返回None
        output: 详细的求解过程字符串
    """
    height = len(maze)
    width = len(maze[0]) if height > 0 else 0

    # 如果没有指定终点，默认为右下角
    if end_pos is None:
        end_pos = (height - 1, width - 1)

    # 确保起点和终点坐标有效
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    if not (0 <= start_x < height and 0 <= start_y < width) or maze[start_x][start_y] == 1:
        return None, "起点坐标无效或被墙壁阻挡"
    if not (0 <= end_x < height and 0 <= end_y < width) or maze[end_x][end_y] == 1:
        return None, "终点坐标无效或被墙壁阻挡"

    # 定义移动方向：右、下、左、上
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_names = ["右", "下", "左", "上"]

    # 使用BFS寻找路径
    queue = deque([(start_x, start_y, [(start_x, start_y)])])  # 队列中存储当前点和路径
    visited = set([(start_x, start_y)])  # 使用集合记录已访问的位置

    output = "开始使用广度优先搜索(BFS)求解迷宫...\n\n"
    output += f"起点: {start_pos}\n"
    output += f"终点: {end_pos}\n\n"

    step = 0

    while queue:
        step += 1
        x, y, path = queue.popleft()

        output += f"步骤 {step}:\n"
        output += f"当前位置: ({x}, {y})\n"

        if (x, y) == end_pos:  # 到达终点
            output += "找到终点！\n"
            output += f"路径长度: {len(path)}\n"
            output += "完整路径:\n"
            output += str(path)
            return path, output

        # 探索四个方向
        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if (0 <= nx < height and 0 <= ny < width and
                    maze[nx][ny] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny, path + [(nx, ny)]))
                output += f"  添加位置 ({nx}, {ny}) 到队列 (向{direction_names[i]})\n"

        output += "\n"

    output += "无法找到从起点到终点的路径。\n"
    return None, output


def is_path_exist(maze, start_pos=(0, 0), end_pos=None):
    """
    检查迷宫是否存在从起点到终点的路径（使用 BFS）

    参数:
        maze: 迷宫的二维数组
        start_pos: 起点位置
        end_pos: 终点位置

    返回:
        exists: 是否存在路径（True/False）
    """
    height = len(maze)
    width = len(maze[0]) if height > 0 else 0

    # 如果没有指定终点，默认为右下角
    if end_pos is None:
        end_pos = (height - 1, width - 1)

    # 确保起点和终点坐标有效
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    if not (0 <= start_x < height and 0 <= start_y < width) or maze[start_x][start_y] == 1:
        return False
    if not (0 <= end_x < height and 0 <= end_y < width) or maze[end_x][end_y] == 1:
        return False

    # 定义移动方向：右、下、左、上
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # 使用BFS寻找路径
    queue = deque([(start_x, start_y)])
    visited = set([(start_x, start_y)])

    while queue:
        x, y = queue.popleft()

        if (x, y) == end_pos:  # 到达终点
            return True

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < height and 0 <= ny < width and
                    maze[nx][ny] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny))

    return False


if __name__ == "__main__":
    # 测试迷宫求解
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    path, output = solve_maze(maze)
    print(output)

    # 测试路径存在性检查
    exists = is_path_exist(maze)
    print(f"路径存在: {exists}")
