import random


def print_grid(grid, stars=None):
    if stars is None:
        stars = []
    for i, row in enumerate(grid):
        row_str = ''
        for j, cell in enumerate(row):
            if (i, j) in stars:
                row_str += '★ '
            else:
                row_str += f'{cell:2} '  # Format the number to take up two spaces
        print(row_str)
    print()

def find_shapes(grid):
    shapes = {}
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell not in shapes:
                shapes[cell] = []
            shapes[cell].append((i, j))
    return shapes

def can_place_star(star_positions, pos):
    row, col = pos

    for star_row, star_col in star_positions:
        if (star_row == row or star_col == col or
            (abs(star_row - row) == abs(star_col - col)) and abs(star_col - col) == 1):
            return False
    return True

def solve_star_battle(grid):
    shapes_dict = find_shapes(grid)
    shapes = [(k,v) for k,v in shapes_dict.items()]
    shape_ids = list(shapes_dict.keys())
    shapes.sort(key=lambda x:len(x[1]))  # Sort by the size of the shapes
    shape_ids = [shape_id for _, shape_id in zip([len(shape) for shape in shapes], shape_ids)]
    
    star_positions = []
    print("初始网格:")
    print_grid(grid)
    attempts = 0
    def backtrack(index=0):
        nonlocal attempts
        shapes.sort(key=lambda x:len(x[1]))  # Sort by the size of the shapes
        if index == len(shapes):
            print("所有区域均已处理完毕")
            return True
        
        current_shape_id = shapes[index][0]
        shape = shapes[index][1]
        print(f"尝试在 {current_shape_id} 区域合法位置填星")
        invalid_positions = []
        for pos in random.sample(shape,len(shape)):
            if can_place_star(star_positions, pos):

                print(f"尝试放置星星在位置 {pos}")
                star_positions.append(pos)
                print(f"当前星星位置：")
                attempts += 1
                print_grid(grid, star_positions)

                if backtrack(index + 1):
                    return True
                
                print(f"移除位置 {pos} 的星星，因为无法在后续区域中找到合适的放置点")
                star_positions.remove(pos)
                print(f"当前星星位置：")
                print_grid(grid, star_positions)
            else:
                # print(f"不能放置星星在位置 {pos}，因为它违反了规则（与已有的星星相邻）")
                invalid_positions.append(pos)
        if invalid_positions:
            print(f"位置{','.join(str(pos) for pos in invalid_positions)}与规则冲突，不能放置星星")
        print(f"当前 {current_shape_id} 区域没有剩余合法的星星放置位置,回溯到 {shapes[index-1][0]} 区域，") if index > 0 else None
        return False
    
    if backtrack():
        print("找到解决方案：")
        print_grid(grid,star_positions)
        print("星星位置：", [ [x,y] for x,y in star_positions])
        return star_positions, attempts
    else:
        print("没有找到解决方案")
        return None,None

if __name__ == "__main__":
    # 示例网格，数字代表所在区域编号
    from grids import grids
    grid = grids[-1]
    star_positions, attempts = solve_star_battle(grid)
    print(f"尝试次数：{attempts}")