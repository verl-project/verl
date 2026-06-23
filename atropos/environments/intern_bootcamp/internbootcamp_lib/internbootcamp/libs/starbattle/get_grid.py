import random

def generate_star_battle_grid(size):
    """
    生成星战谜题的测试用例网格。
    
    参数:
    size -- 网格的大小 (假设是正方形网格，size 表示边长)
    num_regions -- 区域的数量（等于星星数量）
    
    返回:
    grid -- 生成的网格，每个区域由不同的数字表示，0 表示尚未分配区域的空位
    star_positions -- 星星的位置列表
    """
    num_regions = size

    # 初始化网格和可用单元格列表
    grid = [[0 for _ in range(size)] for _ in range(size)]
    available_cells = [(r, c) for r in range(size) for c in range(size)]
    available_cells_for_star = available_cells.copy()
    star_positions = []

    def is_safe_to_place_star(row, col, star_positions):
        """检查在给定位置放置星星是否安全"""
        for r, c in star_positions:
            if row == r or col == c or abs(row - r) == abs(col - c):
                return False
        return True

    while True:
        try:
            # 随机确定所有星星的位置
            while len(star_positions) < num_regions:
                pos = random.choice(available_cells_for_star)
                star_positions.append(pos)
                available_cells_for_star.remove(pos)
                available_cells.remove(pos)
                # pos周围一层也从available_cells_for_star移除
                for r in range(pos[0] - 1, pos[0] ):
                    for c in range(pos[1] - 1, pos[1] ):
                        if (r, c) in available_cells_for_star:
                            available_cells_for_star.remove((r, c))
                # pos 同行同列从available_cells_for_star移除
                for c in range(num_regions):
                    if (pos[0], c) in available_cells_for_star:
                        available_cells_for_star.remove((pos[0], c))
                for r in range(num_regions):
                    if (r, pos[1]) in available_cells_for_star:
                        available_cells_for_star.remove((r, pos[1]))
            break
        except:
            available_cells = [(r, c) for r in range(size) for c in range(size)]
            available_cells_for_star = available_cells.copy()
            star_positions = []
            pass
            
    remaining_cells = list(available_cells)
    # 为每个星星创建一个区域
    regions = []
    for i, star_pos in enumerate(star_positions, start=1):
        region_id = chr(ord('A') + i - 1)
        cur_region,remaining_cells = expand_region_by_1_step(grid, star_pos, remaining_cells, region_id)
        regions.append(cur_region)
    # 继续扩展
    for i,region in enumerate(regions, start=1):
        region_id = chr(ord('A') + i - 1)
        remaining_cells = expand_region_from_star(grid, region, remaining_cells, region_id)

    # 将最后剩余空位随机融合到附近的区域
    def find_nearest_region(grid, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(directions)  # 随机化扩展顺序
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            try:
                if grid[nx][ny] != 0 and nx >= 0 and ny >= 0:
                    return grid[nx][ny]
            except IndexError:
                continue
        return None
    
    while remaining_cells:
        x, y = random.choice(remaining_cells)
        region_id = find_nearest_region(grid, x, y)
        if region_id is not None:
            grid[x][y] = region_id
            remaining_cells.remove((x, y))

    return grid, star_positions

def expand_region_by_1_step(grid,star_pos,available_cells, region_id):
    """从星星位置随机扩散出一个区域"""
    region = [star_pos]
    grid[star_pos[0]][star_pos[1]] = region_id
    # 定义可能的移动方向
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(directions)  # 随机化扩展顺序

    for direction in directions:
        new_pos = (star_pos[0] + direction[0], star_pos[1] + direction[1])
        if new_pos in available_cells:
            available_cells.remove(new_pos)
            region.append(new_pos)
            grid[new_pos[0]][new_pos[1]] = region_id
            break
    return region,available_cells
def expand_region_from_star(grid, region, available_cells, region_id):
    # 继续尝试扩展直到达到合理大小或没有更多可用单元格
    max_region_size = random.randint(1 ,len(available_cells) // len(grid))
    retry = 0
    while available_cells and len(region) < max_region_size and retry < len(grid) ** 2:
        next_cell = random.choice(available_cells)
        if any(abs(next_cell[0] - cell[0]) + abs(next_cell[1] - cell[1]) <= 1 for cell in region):
            available_cells.remove(next_cell)
            region.append(next_cell)
            grid[next_cell[0]][next_cell[1]] = region_id
        retry += 1
    return available_cells

# 示例使用：
if __name__ == "__main__":
    size = 5  # 网格大小

    grid, star_positions = generate_star_battle_grid(size)
    print("生成的网格:")
    for row in grid:
        print(row)
    print("星星的位置:", star_positions)