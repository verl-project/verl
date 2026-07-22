from z3 import *
import numpy as np
import time
import itertools
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def solve_masyu_with_z3(puzzle, timeout=None):
    """
    使用Z3求解器解决Masyu谜题

    参数:
        puzzle: 表示谜题的二维数组或列表，0表示空单元格，1表示白珠，2表示黑珠
        timeout: 求解超时时间（秒）

    返回:
        解决方案路径，超时或无解则返回None
    """
    start_time = time.time()

    # 将列表转换为NumPy数组（如果尚未转换）
    if not isinstance(puzzle, np.ndarray):
        # 打印原始谜题数据以进行调试
        # print("Original puzzle data:", puzzle)

        # 检查谜题格式
        if isinstance(puzzle, list) and all(isinstance(row, list) for row in puzzle):
            # 如果是字符表示的谜题，转换为数字表示
            if isinstance(puzzle[0][0], str):
                # print("Converting character-based puzzle to numeric format...")
                numeric_puzzle = []
                for row in puzzle:
                    numeric_row = []
                    for cell in row:
                        if cell == ' ':
                            numeric_row.append(0)  # 空单元格
                        elif cell == 'W':
                            numeric_row.append(1)  # 白珠
                        elif cell == 'B':
                            numeric_row.append(2)  # 黑珠
                        else:
                            numeric_row.append(0)  # 默认为空
                    numeric_puzzle.append(numeric_row)
                puzzle = np.array(numeric_puzzle)
            else:
                # 标准的二维列表格式
                puzzle = np.array(puzzle)
        else:
            # 尝试解析其他格式
            # print("Attempting to parse non-standard puzzle format...")

            # 如果是字典格式，尝试提取珠子位置
            if isinstance(puzzle, dict):
                # 获取白珠和黑珠的位置
                white_pearls = puzzle.get('white', [])
                black_pearls = puzzle.get('black', [])

                # 确定谜题尺寸
                max_row = max([p[0] for p in white_pearls + black_pearls], default=0) + 1
                max_col = max([p[1] for p in white_pearls + black_pearls], default=0) + 1

                # 创建空谜题
                size = max(max_row, max_col, 4)  # 至少4x4
                puzzle_array = np.zeros((size, size), dtype=int)

                # 放置珠子
                for r, c in white_pearls:
                    puzzle_array[r, c] = 1
                for r, c in black_pearls:
                    puzzle_array[r, c] = 2

                puzzle = puzzle_array
            else:
                # 如果格式无法识别，创建一个默认的空谜题
                # print("Warning: Unrecognized puzzle format. Creating empty 4x4 puzzle.")
                puzzle = np.zeros((4, 4), dtype=int)

    rows, cols = puzzle.shape

    # 打印谜题信息
    # print(f"Puzzle dimensions: {rows}x{cols}")
    # print("Puzzle contents:")
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            if puzzle[r, c] == 0:
                row_str += ". "
            elif puzzle[r, c] == 1:
                row_str += "○ "
            elif puzzle[r, c] == 2:
                row_str += "● "
        # print(row_str)

    # 计算珠子数量
    white_pearls = np.sum(puzzle == 1)
    black_pearls = np.sum(puzzle == 2)
    # print(f"White pearls: {white_pearls}, Black pearls: {black_pearls}")

    # 如果没有珠子，这可能是一个格式问题
    if white_pearls == 0 and black_pearls == 0:
        # print("Warning: No pearls found in the puzzle. This might indicate a format issue.")
        return None

    s = Solver()

    if timeout:
        s.set("timeout", timeout * 1000)  # Z3 timeout in milliseconds

    # 变量定义
    in_path = {}
    for r in range(rows):
        for c in range(cols):
            in_path[(r, c)] = Bool(f"in_path_{r}_{c}")

    # 边变量
    edge = {}
    for r in range(rows):
        for c in range(cols):
            # 水平边
            if c < cols - 1:
                edge[(r, c, r, c + 1)] = Bool(f"edge_{r}_{c}_{r}_{c + 1}")
                edge[(r, c + 1, r, c)] = edge[(r, c, r, c + 1)]  # 双向边
            # 垂直边
            if r < rows - 1:
                edge[(r, c, r + 1, c)] = Bool(f"edge_{r}_{c}_{r + 1}_{c}")
                edge[(r + 1, c, r, c)] = edge[(r, c, r + 1, c)]  # 双向边

    # 约束：路径必须是连续的闭环
    for r in range(rows):
        for c in range(cols):
            # 如果单元格在路径上，它必须恰好有两个相邻的单元格也在路径上
            neighbors = []
            if r > 0:
                neighbors.append((r - 1, c))
            if r < rows - 1:
                neighbors.append((r + 1, c))
            if c > 0:
                neighbors.append((r, c - 1))
            if c < cols - 1:
                neighbors.append((r, c + 1))

            # 如果单元格在路径上，它必须有恰好两个连接的邻居
            connected_neighbors = [edge[(r, c, nr, nc)] for nr, nc in neighbors]

            # 在路径上的单元格必须有恰好两个连接的邻居
            s.add(Implies(in_path[(r, c)], PbEq([(conn, 1) for conn in connected_neighbors], 2)))

            # 如果单元格不在路径上，它不能连接到任何邻居
            s.add(Implies(Not(in_path[(r, c)]), And([Not(edge[(r, c, nr, nc)]) for nr, nc in neighbors])))

            # 确保边的一致性：如果有边连接，则两端的单元格都必须在路径上
            for nr, nc in neighbors:
                s.add(Implies(edge[(r, c, nr, nc)], And(in_path[(r, c)], in_path[(nr, nc)])))

    # 添加珠子约束
    for r in range(rows):
        for c in range(cols):
            if puzzle[r, c] == 1:  # 白珠
                # print(f"Adding white pearl constraints at ({r}, {c})")
                # 白珠必须在路径上
                s.add(in_path[(r, c)])

                # 白珠约束：路径必须在白珠处直行
                horizontal_neighbors = []
                if c > 0:
                    horizontal_neighbors.append((r, c - 1))
                if c < cols - 1:
                    horizontal_neighbors.append((r, c + 1))

                vertical_neighbors = []
                if r > 0:
                    vertical_neighbors.append((r - 1, c))
                if r < rows - 1:
                    vertical_neighbors.append((r + 1, c))

                # 如果水平方向有两个邻居，则路径可以水平穿过
                if len(horizontal_neighbors) == 2:
                    h_straight = And(edge[(r, c, horizontal_neighbors[0][0], horizontal_neighbors[0][1])],
                                     edge[(r, c, horizontal_neighbors[1][0], horizontal_neighbors[1][1])])
                else:
                    h_straight = False

                # 如果垂直方向有两个邻居，则路径可以垂直穿过
                if len(vertical_neighbors) == 2:
                    v_straight = And(edge[(r, c, vertical_neighbors[0][0], vertical_neighbors[0][1])],
                                     edge[(r, c, vertical_neighbors[1][0], vertical_neighbors[1][1])])
                else:
                    v_straight = False

                # 路径必须在白珠处直行（水平或垂直）
                valid_straights = [x for x in [h_straight, v_straight] if x is not False]
                if valid_straights:
                    s.add(Or(*valid_straights))

                # 添加白珠转弯约束（至少一侧必须转弯）
                if h_straight is not False:
                    # 检查左侧是否可以转弯
                    left_r, left_c = horizontal_neighbors[0]
                    left_turns = []
                    if left_r > 0:
                        left_turns.append(edge[(left_r, left_c, left_r - 1, left_c)])
                    if left_r < rows - 1:
                        left_turns.append(edge[(left_r, left_c, left_r + 1, left_c)])

                    # 检查右侧是否可以转弯
                    right_r, right_c = horizontal_neighbors[1]
                    right_turns = []
                    if right_r > 0:
                        right_turns.append(edge[(right_r, right_c, right_r - 1, right_c)])
                    if right_r < rows - 1:
                        right_turns.append(edge[(right_r, right_c, right_r + 1, right_c)])

                    # 至少一侧必须转弯
                    all_turns = left_turns + right_turns
                    if all_turns:
                        s.add(Implies(h_straight, Or(*all_turns)))

                if v_straight is not False:
                    # 检查上侧是否可以转弯
                    top_r, top_c = vertical_neighbors[0]
                    top_turns = []
                    if top_c > 0:
                        top_turns.append(edge[(top_r, top_c, top_r, top_c - 1)])
                    if top_c < cols - 1:
                        top_turns.append(edge[(top_r, top_c, top_r, top_c + 1)])

                    # 检查下侧是否可以转弯
                    bottom_r, bottom_c = vertical_neighbors[1]
                    bottom_turns = []
                    if bottom_c > 0:
                        bottom_turns.append(edge[(bottom_r, bottom_c, bottom_r, bottom_c - 1)])
                    if bottom_c < cols - 1:
                        bottom_turns.append(edge[(bottom_r, bottom_c, bottom_r, bottom_c + 1)])

                    # 至少一侧必须转弯
                    all_turns = top_turns + bottom_turns
                    if all_turns:
                        s.add(Implies(v_straight, Or(*all_turns)))

            elif puzzle[r, c] == 2:  # 黑珠
                # print(f"Adding black pearl constraints at ({r}, {c})")
                # 黑珠必须在路径上
                s.add(in_path[(r, c)])

                # 获取四个方向的邻居
                neighbors = []
                directions = []  # 存储方向向量

                if r > 0:
                    neighbors.append((r - 1, c))
                    directions.append((-1, 0))

                if r < rows - 1:
                    neighbors.append((r + 1, c))
                    directions.append((1, 0))

                if c > 0:
                    neighbors.append((r, c - 1))
                    directions.append((0, -1))

                if c < cols - 1:
                    neighbors.append((r, c + 1))
                    directions.append((0, 1))

                # 黑珠必须连接到恰好两个邻居
                connected_neighbors = [edge[(r, c, nr, nc)] for nr, nc in neighbors]
                s.add(PbEq([(conn, 1) for conn in connected_neighbors], 2))

                # 黑珠约束1：路径必须在黑珠处转弯
                for i, (nr1, nc1) in enumerate(neighbors):
                    for j, (nr2, nc2) in enumerate(neighbors):
                        if i < j:
                            if (nr1 == r and nr2 == r and nc1 != nc2) or (nc1 == c and nc2 == c and nr1 != nr2):
                                s.add(Not(And(edge[(r, c, nr1, nc1)], edge[(r, c, nr2, nc2)])))

                # 黑珠约束2：路径必须在黑珠两侧至少直行一格
                for i, ((nr, nc), (dr, dc)) in enumerate(zip(neighbors, directions)):
                    next_r, next_c = nr + dr, nc + dc
                    if 0 <= next_r < rows and 0 <= next_c < cols:
                        s.add(Implies(
                            edge[(r, c, nr, nc)],
                            edge[(nr, nc, next_r, next_c)]
                        ))

    # 添加约束：至少有一个单元格在路径上
    s.add(Or([in_path[(r, c)] for r in range(rows) for c in range(cols)]))

    # 添加子路径消除约束
    # 使用一个辅助变量表示单元格的访问顺序
    visit_order = {}
    for r in range(rows):
        for c in range(cols):
            visit_order[(r, c)] = Int(f"visit_{r}_{c}")
            # 如果单元格不在路径上，其访问顺序为0
            s.add(Implies(Not(in_path[(r, c)]), visit_order[(r, c)] == 0))
            # 如果单元格在路径上，其访问顺序为正数
            s.add(Implies(in_path[(r, c)], visit_order[(r, c)] > 0))
            # 访问顺序不能超过单元格总数
            s.add(visit_order[(r, c)] <= rows * cols)

    # 确保访问顺序形成一个有效的序列
    for r in range(rows):
        for c in range(cols):
            neighbors = []
            if r > 0:
                neighbors.append((r - 1, c))
            if r < rows - 1:
                neighbors.append((r + 1, c))
            if c > 0:
                neighbors.append((r, c - 1))
            if c < cols - 1:
                neighbors.append((r, c + 1))

            # 如果单元格在路径上，它的访问顺序必须比其中一个连接的邻居大1，比另一个小1
            # 或者是首尾连接的特殊情况
            neighbor_constraints = []
            for nr, nc in neighbors:
                neighbor_constraints.append(
                    And(
                        edge[(r, c, nr, nc)],
                        Or(
                            visit_order[(r, c)] == visit_order[(nr, nc)] + 1,
                            visit_order[(r, c)] == visit_order[(nr, nc)] - 1,
                            # 特殊情况：处理环的首尾连接
                            And(
                                visit_order[(r, c)] == 1,
                                visit_order[(nr, nc)] == rows * cols
                            ),
                            And(
                                visit_order[(r, c)] == rows * cols,
                                visit_order[(nr, nc)] == 1
                            )
                        )
                    )
                )

            # 如果单元格在路径上，必须满足至少两个邻居约束
            s.add(Implies(
                in_path[(r, c)],
                PbEq([(constraint, 1) for constraint in neighbor_constraints], 2)
            ))

    # 求解
    # print("Starting Z3 solver...")
    result = s.check()
    solve_time = time.time() - start_time
    # print(f"Z3 result: {result}")

    if result == sat:
        model = s.model()
        # print("Solution found!")

        # 构建有序路径
        path_with_order = []
        for r in range(rows):
            for c in range(cols):
                if model.evaluate(in_path[(r, c)]):
                    order = model.evaluate(visit_order[(r, c)]).as_long()
                    if order > 0:  # 只考虑在路径上的单元格
                        path_with_order.append((order, (r, c)))

        # 按访问顺序排序
        path_with_order.sort()
        ordered_path = [pos for _, pos in path_with_order]

        # 确保路径是闭环
        if ordered_path and ordered_path[0] != ordered_path[-1]:
            ordered_path.append(ordered_path[0])

            # 验证路径连续性
            if not is_continuous_path(ordered_path):
                # print("Warning: Z3 solution is not a continuous path. Attempting to fix...")
                fixed_path = construct_continuous_path(ordered_path, model, edge, rows, cols)
                if fixed_path:
                    ordered_path = fixed_path
                    # print(f"Fixed path: {ordered_path}")
                else:
                    # print("Failed to fix path. Using fallback solver...")
                    ordered_path = solve_with_backtracking(puzzle)

            # 验证路径是否满足所有规则
            if ordered_path and not verify_complete_path(puzzle, ordered_path):
                # print("Warning: Z3 solution does not satisfy all rules. Using fallback solver...")
                ordered_path = solve_with_backtracking(puzzle)

            # print(f"Final solution path: {ordered_path}")
            # print(f"Solved in {solve_time:.2f} seconds.")
            return ordered_path

    elif result == unknown:
        # print(f"Z3 timed out after {solve_time:.2f} seconds")
        # print("Using fallback backtracking solver...")
        return solve_with_backtracking(puzzle)
    else:
        # print(f"Z3 determined puzzle is unsatisfiable in {solve_time:.2f} seconds")
        return None


def is_continuous_path(path):
    """检查路径是否连续"""
    if not path or len(path) < 2:
        return False

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        if not ((abs(r1 - r2) == 1 and c1 == c2) or (r1 == r2 and abs(c1 - c2) == 1)):
            # print(f"Path discontinuity at {path[i]} to {path[i + 1]}")
            return False

    return True


def construct_continuous_path(cells, model, edge, rows, cols):
    """
    从Z3解中构建连续路径

    参数:
        cells: 路径上的单元格集合
        model: Z3模型
        edge: 边变量字典
        rows, cols: 网格尺寸

    返回:
        连续路径
    """
    if not cells:
        return None

    # 构建邻接图
    graph = {cell: [] for cell in cells}
    for r1 in range(rows):
        for c1 in range(cols):
            for dr, dc in [(0, 1), (1, 0)]:
                r2, c2 = r1 + dr, c1 + dc
                if r2 < rows and c2 < cols:
                    if (r1, c1) in graph and (r2, c2) in graph:
                        if model.evaluate(edge[(r1, c1, r2, c2)]):
                            graph[(r1, c1)].append((r2, c2))
                            graph[(r2, c2)].append((r1, c1))

    # 从第一个单元格开始DFS
    start = cells[0]
    path = [start]
    visited = {start}

    # DFS构建路径
    current = start
    while len(visited) < len(cells):
        found_next = False
        for neighbor in graph[current]:
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)
                current = neighbor
                found_next = True
                break

        if not found_next:
            # 如果无法继续，尝试从其他已访问点继续
            for i, node in enumerate(path):
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        # 重新排序路径
                        new_path = path[i:] + path[:i]
                        path = new_path
                        current = node
                        found_next = True
                        break
                if found_next:
                    break

            # 如果仍然无法继续，可能是无效解
            if not found_next:
                return None

    # 检查是否可以闭合路径
    if start in graph[current]:
        path.append(start)  # 闭合路径

    return path


def solve_with_backtracking(puzzle):
    """
    使用回溯算法解决Masyu谜题

    参数:
        puzzle: 谜题网格

    返回:
        solution_path: 解决方案路径，如果无解则返回None
    """
    # print("Using improved backtracking algorithm to solve Masyu puzzle...")

    rows, cols = puzzle.shape

    # 获取所有珠子位置
    pearls = []
    for r in range(rows):
        for c in range(cols):
            if puzzle[r, c] in [1, 2]:  # 白珠或黑珠
                pearls.append((r, c))

    # 如果没有珠子，返回None
    if not pearls:
        return None

    # 尝试从每个珠子开始
    for start in pearls:
        # 初始化路径和访问状态
        path = [start]
        visited = {start}

        # 尝试构建路径
        if improved_backtrack(puzzle, path, visited, pearls, start):
            # 确保路径是闭环
            if path[0] != path[-1]:
                path.append(path[0])
            return path

    return None


def improved_backtrack(puzzle, path, visited, pearls, start_pos):
    """
    改进的回溯算法构建路径

    参数:
        puzzle: 谜题网格
        path: 当前路径
        visited: 已访问的单元格
        pearls: 所有珠子位置
        start_pos: 起始位置

    返回:
        是否找到有效路径
    """
    # 如果已经访问了所有珠子，尝试闭合路径
    if all(p in visited for p in pearls):
        # 检查是否可以回到起点
        r, c = path[-1]
        start_r, start_c = start_pos

        # 如果可以直接回到起点，路径有效
        if abs(r - start_r) + abs(c - start_c) == 1:
            # 验证整个路径是否满足所有规则
            if verify_complete_path(puzzle, path + [start_pos]):
                return True

        return False

    # 获取当前位置
    r, c = path[-1]
    rows, cols = puzzle.shape

    # 尝试四个方向
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # 如果当前位置是珠子，需要特殊处理方向优先级
    if puzzle[r, c] == 1:  # 白珠
        # 如果路径长度大于1，确保直行
        if len(path) > 1:
            prev_r, prev_c = path[-2]
            dr, dc = r - prev_r, c - prev_c
            # 优先直行方向
            directions = [(dr, dc)] + [d for d in directions if d != (dr, dc)]
    elif puzzle[r, c] == 2:  # 黑珠
        # 如果路径长度大于1，确保转弯
        if len(path) > 1:
            prev_r, prev_c = path[-2]
            dr, dc = r - prev_r, c - prev_c
            # 优先转弯方向
            turn_directions = [(-dc, dr), (dc, -dr)]
            directions = turn_directions + [d for d in directions if d not in turn_directions]

    for dr, dc in directions:
        nr, nc = r + dr, c + dc

        # 检查是否在网格内且未访问
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
            # 检查移动是否符合规则
            if improved_is_valid_move(puzzle, path, r, c, nr, nc):
                # 添加到路径
                path.append((nr, nc))
                visited.add((nr, nc))

                # 递归搜索
                if improved_backtrack(puzzle, path, visited, pearls, start_pos):
                    return True

                # 回溯
                path.pop()
                visited.remove((nr, nc))

    return False


def improved_is_valid_move(puzzle, path, r1, c1, r2, c2):
    """
    改进的移动有效性检查

    参数:
        puzzle: 谜题网格
        path: 当前路径
        r1, c1: 当前位置
        r2, c2: 目标位置

    返回:
        移动是否有效
    """
    rows, cols = puzzle.shape

    # 检查白珠规则
    if puzzle[r1, c1] == 1:  # 白珠
        # 获取前一个位置（如果有）
        if len(path) >= 2:
            prev_r, prev_c = path[-2]

            # 检查是否直行
            if not ((prev_r == r1 == r2) or (prev_c == c1 == c2)):
                return False

    # 检查黑珠规则
    if puzzle[r1, c1] == 2:  # 黑珠
        # 获取前一个位置（如果有）
        if len(path) >= 2:
            prev_r, prev_c = path[-2]

            # 检查是否转弯
            if (prev_r == r1 == r2) or (prev_c == c1 == c2):
                return False

            # 检查后方是否可以直行
            dr, dc = r2 - r1, c2 - c1
            next_r, next_c = r2 + dr, c2 + dc

            # 检查下一个位置是否在网格内
            if not (0 <= next_r < rows and 0 <= next_c < cols):
                return False

    # 检查目标位置是否为白珠
    if puzzle[r2, c2] == 1:  # 白珠
        # 白珠必须能够直行通过
        dr, dc = r2 - r1, c2 - c1
        next_r, next_c = r2 + dr, c2 + dc

        # 检查下一个位置是否在网格内
        if not (0 <= next_r < rows and 0 <= next_c < cols):
            return False

    # 检查目标位置是否为黑珠
    if puzzle[r2, c2] == 2:  # 黑珠
        # 黑珠必须能够转弯
        dr, dc = r2 - r1, c2 - c1

        # 可能的转弯方向
        turn_directions = [(-dc, dr), (dc, -dr)]

        valid_turn = False
        for turn_dr, turn_dc in turn_directions:
            next_r, next_c = r2 + turn_dr, c2 + turn_dc

            # 检查转弯后的位置是否在网格内
            if 0 <= next_r < rows and 0 <= next_c < cols:
                valid_turn = True
                break

        if not valid_turn:
            return False

    return True


def verify_complete_path(puzzle, path):
    """
    验证完整路径是否满足所有规则

    参数:
        puzzle: 谜题网格
        path: 完整路径

    返回:
        路径是否有效
    """
    rows, cols = puzzle.shape

    # 检查白珠规则
    for r in range(rows):
        for c in range(cols):
            if puzzle[r, c] == 1:  # 白珠
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

                if not (prev_turn or next_turn):
                    return False

    # 检查黑珠规则
    for r in range(rows):
        for c in range(cols):
            if puzzle[r, c] == 2:  # 黑珠
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

                if not (prev_straight and next_straight):
                    return False

    return True

