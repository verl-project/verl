def check_valid_masyu(grid):
    """
    检查Masyu谜题是否有效

    参数:
        grid: 谜题网格

    返回:
        is_valid: 谜题是否有效
    """
    # 简化的验证逻辑，主要检查珠子的位置是否合理
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # 检查网格大小
    if rows < 4 or cols < 4:
        return False

    # 计算珠子数量
    black_pearls = sum(row.count('B') for row in grid)
    white_pearls = sum(row.count('W') for row in grid)

    # 检查珠子数量是否合理
    if black_pearls == 0 and white_pearls == 0:
        return False

    # 检查珠子是否有足够的空间
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'B':
                # 黑珠需要有足够的空间来满足规则
                # 检查是否有至少两个方向可以直行
                valid_directions = 0

                # 检查上方
                if r >= 2:
                    valid_directions += 1

                # 检查下方
                if r <= rows - 3:
                    valid_directions += 1

                # 检查左方
                if c >= 2:
                    valid_directions += 1

                # 检查右方
                if c <= cols - 3:
                    valid_directions += 1

                # 黑珠至少需要两个有效方向
                if valid_directions < 2:
                    return False

            elif grid[r][c] == 'W':
                # 白珠需要有足够的空间来满足规则
                # 检查是否有至少一个方向可以直行并拐弯
                valid_directions = 0

                # 检查水平方向
                if c >= 1 and c <= cols - 2:
                    valid_directions += 1

                # 检查垂直方向
                if r >= 1 and r <= rows - 2:
                    valid_directions += 1

                # 白珠至少需要一个有效方向
                if valid_directions < 1:
                    return False

    # 通过所有检查，谜题有效
    return True
