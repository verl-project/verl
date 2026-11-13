import random
from typing import List, Tuple, Optional
from collections import deque


def generate_maze(width: int, height: int, start_pos=(0, 0), end_pos=None, difficulty=1, seed=None) -> List[List[int]]:
    """
    优化后的随机迷宫生成器

    参数:
        width: 迷宫宽度（列数）
        height: 迷宫高度（行数）
        start_pos: 起点位置
        end_pos: 终点位置
        difficulty: 难度（1-3）
        seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)

    # 初始化迷宫
    maze = [[1 for _ in range(width)] for _ in range(height)]

    if end_pos is None:
        end_pos = (height - 1, width - 1)

    start_x, start_y = start_pos
    end_x, end_y = end_pos

    # 验证起点和终点坐标
    if not (0 <= start_x < width and 0 <= start_y < height):
        raise ValueError("起点坐标无效")
    if not (0 <= end_x < width and 0 <= end_y < height):
        raise ValueError("终点坐标无效")

    # 增加更多方向选择，包括对角线移动用于生成过程
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # 基本方向
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线方向
    ]

    visited = [[False for _ in range(width)] for _ in range(height)]

    # 使用改进的生成算法
    def carve_passages(x: int, y: int, branch_chance: float):
        visited[y][x] = True
        maze[y][x] = 0

        # 获取随机打乱的方向
        current_dirs = directions[:4] if random.random() > 0.3 else directions  # 30%概率使用对角线
        random.shuffle(current_dirs)

        for dx, dy in current_dirs:
            nx, ny = x + dx * 2, y + dy * 2

            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                # 根据难度和分支概率决定是否开辟新路径
                if random.random() < branch_chance:
                    # 在两点之间开通路径
                    maze[y + dy][x + dx] = 0
                    maze[ny][nx] = 0
                    carve_passages(nx, ny, branch_chance * 0.95)  # 随深度递减分支概率

                    # 随机创建环路
                    if random.random() < 0.2:  # 20%概率创建环路
                        for loop_dx, loop_dy in random.sample(directions[:4], 2):  # 只使用基本方向创建环路
                            loop_x, loop_y = nx + loop_dx, ny + loop_dy
                            if (0 <= loop_x < width and 0 <= loop_y < height and
                                    maze[loop_y][loop_x] == 0):
                                maze[ny][nx] = 0
                                break

    # 根据难度调整分支概率
    initial_branch_chance = {
        1: 0.75,  # 简单：较少分支
        2: 0.85,  # 中等：适中分支
        3: 0.95  # 困难：大量分支
    }.get(difficulty, 0.85)

    # 从起点开始生成迷宫
    carve_passages(start_x, start_y, initial_branch_chance)

    # 添加随机特征
    add_random_features(maze, difficulty, start_pos, end_pos)

    # 确保起点和终点是通路
    maze[start_y][start_x] = 0
    maze[end_y][end_x] = 0

    # 如果没有路径，创建一条
    if not has_path(maze, start_pos, end_pos):
        create_enhanced_path(maze, start_pos, end_pos)

    return maze


def add_random_features(maze: List[List[int]], difficulty: int, start_pos: Tuple[int, int],
                        end_pos: Tuple[int, int]):
    """添加随机特征以增加迷宫的复杂性和随机性"""
    height, width = len(maze), len(maze[0])

    # 根据难度确定要添加的随机特征数量
    feature_count = difficulty * (width + height) // 4

    for _ in range(feature_count):
        feature_type = random.random()

        if feature_type < 0.4:  # 40%概率：添加或移除单个墙
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if (y, x) != start_pos and (y, x) != end_pos:
                maze[y][x] = 1 if random.random() < 0.5 else 0

        elif feature_type < 0.7:  # 30%概率：创建小房间
            room_width = random.randint(2, 4)
            room_height = random.randint(2, 4)
            x = random.randint(1, width - room_width - 1)
            y = random.randint(1, height - room_height - 1)

            # 确保房间不会覆盖起点或终点
            if not (start_pos[1] >= y and start_pos[1] < y + room_height and
                    start_pos[0] >= x and start_pos[0] < x + room_width) and \
                    not (end_pos[1] >= y and end_pos[1] < y + room_height and
                         end_pos[0] >= x and end_pos[0] < x + room_width):
                for i in range(room_height):
                    for j in range(room_width):
                        maze[y + i][x + j] = 0

        else:  # 30%概率：创建通道
            if random.random() < 0.5:  # 水平通道
                y = random.randint(1, height - 2)
                for x in range(1, width - 1):
                    if (y, x) != start_pos and (y, x) != end_pos:
                        maze[y][x] = 0
            else:  # 垂直通道
                x = random.randint(1, width - 2)
                for y in range(1, height - 1):
                    if (y, x) != start_pos and (y, x) != end_pos:
                        maze[y][x] = 0


def has_path(maze: List[List[int]], start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> bool:
    """
    检查迷宫是否有从起点到终点的路径

    参数:
        maze: 迷宫二维数组
        start_pos: 起点坐标 (x, y)
        end_pos: 终点坐标 (x, y)

    返回:
        bool: 是否存在可行路径
    """
    width, height = len(maze[0]), len(maze)
    visited = [[False for _ in range(width)] for _ in range(height)]
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    queue = deque([(start_x, start_y)])
    visited[start_y][start_x] = True

    # 四个基本方向：右、下、左、上
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        x, y = queue.popleft()

        if (x, y) == (end_x, end_y):
            return True

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height and
                    maze[ny][nx] == 0 and not visited[ny][nx]):
                visited[ny][nx] = True
                queue.append((nx, ny))

    return False


def create_enhanced_path(maze: List[List[int]], start_pos: Tuple[int, int],
                         end_pos: Tuple[int, int]):
    """创建一条更自然的路径从起点到终点"""
    start_x, start_y = start_pos
    end_x, end_y = end_pos
    current_x, current_y = start_x, start_y

    while (current_x, current_y) != (end_x, end_y):
        # 随机选择是先移动x还是y
        if random.random() < 0.5:
            if current_x != end_x:
                current_x += 1 if current_x < end_x else -1
                maze[current_y][current_x] = 0
            elif current_y != end_y:
                current_y += 1 if current_y < end_y else -1
                maze[current_y][current_x] = 0
        else:
            if current_y != end_y:
                current_y += 1 if current_y < end_y else -1
                maze[current_y][current_x] = 0
            elif current_x != end_x:
                current_x += 1 if current_x < end_x else -1
                maze[current_y][current_x] = 0

        # 随机添加一些支路
        if random.random() < 0.2:  # 20%概率添加支路
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            nx, ny = current_x + dx, current_y + dy
            if (0 <= nx < len(maze[0]) and 0 <= ny < len(maze) and
                    (ny, nx) != start_pos and (ny, nx) != end_pos):
                maze[ny][nx] = 0


def print_maze(maze: List[List[int]]):
    """
    打印迷宫（用于调试）

    参数:
        maze: 迷宫的二维数组
    """
    for row in maze:
        print("".join(["  " if cell == 0 else "██" for cell in row]))


if __name__ == "__main__":
    # 测试迷宫生成
    maze = generate_maze(15, 15, difficulty=2)
    print_maze(maze)
