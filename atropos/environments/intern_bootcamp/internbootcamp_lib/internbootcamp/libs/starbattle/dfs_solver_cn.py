import random
from .dfs_solver_en import print_grid_in_kor

def print_grid(grid, stars=None):
    if stars is None:
        stars = []
    row_str_list = []
    for i, row in enumerate(grid):
        row_str = f'['
        for j, cell in enumerate(row):
            if (i, j) in stars:
                row_str += '*, ' if j + 1 < len(row) else  '*'
            else:
                row_str += f'{cell}, ' if j + 1 < len(row) else  f'{cell}'# Format the number to take up two spaces
        row_str += ']'
        row_str_list.append(row_str)
    # grid_str += "\n"
    return "```starmatrix\n[\n" + ',\n'.join(row_str_list) + "\n]\n```"

def print_stars(stars,grid):
    star_list = []
    stars.sort(key=lambda star: grid[star[0]][star[1]])
    for star in stars:
        star_list.append(f"{grid[star[0]][star[1]]}({star[0] + 1},{star[1] + 1})")
    return "[[" +',\n'.join(star_list)+ "]]"

def find_shapes(grid):
    shapes = {}
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell not in shapes:
                shapes[cell] = []
            shapes[cell].append((i, j))
    return shapes

def can_place_star(star_positions, pos):
    row, col = pos  # 解构位置为行(row)和列(col)
    conflict_info = None

    for star_row, star_col in star_positions:  # 遍历已经放置的星星的位置
        if star_row == row:
            conflict_info = ('在同一行', (star_row, star_col))
            break
        elif star_col == col:
            conflict_info = ('在同一列', (star_row, star_col))
            break
        elif abs(star_row - row) == abs(star_col - col):
            if abs(star_col - col) == 1:
                conflict_info = ('对角线相邻', (star_row, star_col))
                break

    if conflict_info is not None:
        return False, conflict_info  # 返回False以及冲突的信息
    else:
        return True, None  # 如果没有违反任何规则，则可以放置星星，且无冲突信息

def update_valid_positions(shapes, star_positions):
    updated_shapes = []
    for shape_id, positions in shapes:
        valid_positions = [pos for pos in positions if can_place_star(star_positions, pos)]
        updated_shapes.append((shape_id, valid_positions))
    updated_shapes.sort(key=lambda x: len(x[1]))
    return updated_shapes

def solve_star_battle(grid):
    original_grid = grid
    shapes_dict = find_shapes(grid)
    shapes = [(k, v) for k, v in shapes_dict.items()]
    shapes.sort(key=lambda x: len(x[1]))  # 按区域大小排序
    star_positions = []
    
    print("这是一个5x5的星战逻辑解谜游戏。我们需要在网格中放置星星，满足以下条件：\n1.任意两颗星星不能在同一列、同一行或在对角线上相邻\n2.每行、每列和每个标记区域必须放置1颗星星\n3.网格由字母标记区域，同一个字母属于同一个区域\n")
    print("解决这类问题需要：\n1.理解区域划分：相同字母属于同一区域\n2.排除法：通过已知星星位置排除不可能的位置\n3.唯一性：每个行、列、区域只能放一颗星星\n4.邻近规则：星星不能相邻（包括对角线）\n5.回溯法: 逐步构建解决方案并在发现当前路径不可行时“回溯”到上一步以尝试其他可能来探索所有潜在的解决方案，直到找到正确的答案或遍历完所有可能性。\n6.MRV策略:最少剩余值策略，每次处理可能值最少的情况即处理合法位置最少的区域，从而尽可能减少搜索空间。\n" )
    print("先拟定一个解题计划如下：")
    print("1. 识别所有区域，并收集每个区域的格子位置。\n2. 按区域合法位置数排序，每次优先处理合法位置最少的区域。\n3. 在每个区域内尝试放置星星，检查是否符合规则。\n4. 放置星星后重新计算各区域合法位置数，继续选择拥有最少合法位置的区域填星\n5. 使用回溯算法，如果当前区域无法放置，则回溯到上一步，尝试其他位置。\n6. 记录已放置的星星位置，确保不违反规则。\n7. 如果所有区域都处理完毕，输出解决方案。如果搜索完成后没有找到解，则该题无解。\n")
    print("现在开始尝试解题：")
    shape_str = ''
    for shape in [(k, v) for k, v in shapes_dict.items()]:
        shape_str += f"{shape[0]} 区域, 包含格子:{', '.join(f'{x[0] + 1,x[1] + 1}' for x in shape[1])}, 合法位置数: {len(shape[1])}\n"
    
    print("初始网格:\n" + print_grid_in_kor(grid) + "\n识别所有区域，并收集每个区域的格子位置\n" + shape_str + "下面从合法位置最少的区域开始填星，用符号 * 标记已放置的星星位置\n")
    
    
    attempts = 0

    def backtrack(shapes, last_shape_id):
        nonlocal attempts
        if not shapes:
            print("所有区域均填星成功，找到了一个满足所有条件的星星放置方案，具体位置如下：\n" + print_grid_in_kor(grid, star_positions) + "\n最终答案为：" + print_stars(star_positions,original_grid))
            return True

        current_shape_id, shape = shapes[0]
        invalid_positions = []
        for pos in random.sample(shape, len(shape)):
            place_res, info= can_place_star(star_positions, pos)
            if place_res:
                print(f"在拥有最少合法位置的 {current_shape_id} 区域填星。\n" + f"尝试放置星星在位置 {(pos[0]+1,pos[1]+1)}\n")
                remaining_shapes = update_valid_positions(shapes[1:], star_positions)
                if len(remaining_shapes) > 0:
                    print(f"位置 {(pos[0]+1,pos[1]+1)} 所在行、列及对角均无其余星星，符合规则。 星星位置已更新，重新计算剩余区域的合法位置数，继续选择拥有最少合法位置的 {remaining_shapes[0][0]} 区域填星\n")
                else:
                    print(f"位置 {(pos[0]+1,pos[1]+1)} 所在行、列及对角均无其余星星，符合规则。 已无剩余区域，看来我们找到到一个满足所有条件的星星放置方案。\n")
                star_positions.append(pos)
                print(f"当前星星位置：{print_stars(star_positions,original_grid)}\n" + "当前网格：\n" + print_grid_in_kor(grid, star_positions))
                attempts += 1

                
                if backtrack(remaining_shapes, last_shape_id=current_shape_id):
                    return True

                print(f"移除位置 {(pos[0]+1,pos[1]+1)} 的星星，因为无法在后续区域中找到合适的放置点")
                star_positions.remove(pos)
                print(f"当前星星位置：{print_stars(star_positions,original_grid)}\n" + "当前网格：\n" + print_grid_in_kor(grid, star_positions))
            else:
                if random.random() < 0.2 and info[0] == '对角线相邻':
                    print(f"在拥有最少合法位置的 {current_shape_id} 区域填星。\n" + f"尝试放置星星在位置 {(pos[0]+1,pos[1]+1)}\n")
                    print(f'{(pos[0]+1,pos[1]+1)} 不合法，因为它与星星 {(info[1][0] + 1, info[1][1] + 1)} {info[0]}。再次尝试在{current_shape_id} 区域放置星星\n')
                    attempts += 1
                elif random.random() < 0.1 and not info[0] == '对角线相邻':
                    print(f"在拥有最少合法位置的 {current_shape_id} 区域填星。\n" + f"尝试放置星星在位置 {(pos[0]+1,pos[1]+1)}\n")
                    print(f'{(pos[0]+1,pos[1]+1)} 不合法，因为它与星星 {(info[1][0] + 1, info[1][1] + 1)} {info[0]}。再次尝试在{current_shape_id} 区域放置星星\n')
                    attempts += 1
                    
                invalid_positions.append(pos)

        if invalid_positions:
            print(f"{current_shape_id} 区域所有位置{','.join(str(pos) for pos in invalid_positions)}与现有星星规则冲突，无合法位置，不能放置星星\n")
        if last_shape_id:
            print(f"当前 {current_shape_id} 区域没有剩余合法的星星放置位置,回溯到 {last_shape_id} 区域\n")
        else:
            # print_conclude("没有找到解决方案")
            pass
        return False

    if backtrack(shapes, None):
        return print_stars(stars=star_positions,grid=original_grid), attempts
    else:
        print("我们已经尝试了所有可能情况,没有找到解决方案。该问题无解。\n")
        return None, None


if __name__ == "__main__":
    # 示例网格，数字代表所在区域编号
    from grids import grids
    grid = grids[-2]
    star_positions, attempts = solve_star_battle(grid)
    print(f"尝试次数：{attempts}")