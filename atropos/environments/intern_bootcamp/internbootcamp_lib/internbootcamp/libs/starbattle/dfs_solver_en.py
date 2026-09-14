import random
import copy

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

def print_grid_in_kor(grid, stars=None):
    grid_cp = copy.deepcopy(grid)
    if stars:
        for i,j in stars:
            grid_cp[i][j] = '*'
            pass
    return '\n'.join(['\t'.join(row) for row in grid_cp])

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
    row, col = pos  # Deconstruct position into row and column
    conflict_info = None

    for star_row, star_col in star_positions:  # Traverse the positions of already placed stars
        if star_row == row:
            conflict_info = ('in the same row', (star_row, star_col))
            break
        elif star_col == col:
            conflict_info = ('in the same column', (star_row, star_col))
            break
        elif abs(star_row - row) == abs(star_col - col):
            if abs(star_col - col) == 1:
                conflict_info = ('diagonally adjacent', (star_row, star_col))
                break

    if conflict_info is not None:
        return False, conflict_info  # Return False and conflict information
    else:
        return True, None  # If no rules are violated, the star can be placed, and there is no conflict information

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
    shapes.sort(key=lambda x: len(x[1]))  # Sort by region size
    star_positions = []
    
    print("This is a 5x5 Star Battle logic puzzle. We need to place stars in the grid, satisfying the following conditions:\n1. Any two stars cannot be in the same row, column, or diagonally adjacent\n2. Each row, column, and marked region must contain exactly one star\n3. The grid is divided into regions marked by letters, and the same letter belongs to the same region.\n")
    print("Solving this type of problem requires:\n1. Understanding region division: the same letter belongs to the same region\n2. Elimination method: exclude impossible positions based on known star positions\n3. Uniqueness: each row, column, and region can only contain one star\n4. Adjacency rule: stars cannot be adjacent (including diagonally)\n5. Backtracking: gradually build the solution and 'backtrack' to the previous step when the current path is found to be infeasible to try other possibilities, exploring all potential solutions until the correct answer is found or all possibilities are exhausted.\n6. MRV strategy: Minimum Remaining Values strategy, each time dealing with the situation with the fewest possible values, i.e., the region with the fewest legal positions, thereby reducing the search space as much as possible.\n")
    print("Here is the step-by-step process of solving this type of problem:\n")
    print("1. Identify all regions and collect the positions of each region.\n2. Sort regions by the number of legal positions, each time prioritizing the region with the fewest legal positions.\n3. Try to place a star in each region, checking if it complies with the rules.\n4. After placing a star, recalculate the number of legal positions in each region, continue to select the region with the fewest legal positions to place a star\n5. Use the backtracking algorithm, if the current region cannot be placed, backtrack to the previous step and try other positions.\n6. Record the positions of placed stars to ensure no rules are violated.\n7. If all regions are processed, output the solution. If no solution is found after the search is completed, the puzzle has no solution.\n")    
    shape_str = ''
    for shape in [(k, v) for k, v in shapes_dict.items()]:
        shape_str += f"region {shape[0]}, contains cells: {', '.join(f'{x[0] + 1,x[1] + 1}' for x in shape[1])}, number of legal positions: {len(shape[1])}\n"
    
    print("Try to solve the puzzle:")
    print("Initial grid:\n" + print_grid_in_kor(grid) + "\nIdentify all regions and collect the positions of each region\n" + shape_str + "Now, start placing stars from the region with the fewest legal positions, using the symbol * to mark the positions of placed stars.")
    
    
    attempts = 0

    def backtrack(shapes, last_shape_id):
        nonlocal attempts
        if not shapes:
            print("All regions have been successfully filled with stars, a star placement solution that satisfies all conditions has been found, the specific positions are as follows:\n" + print_grid_in_kor(grid, star_positions) + "\nThe final answer is:\n" + print_stars(star_positions,original_grid))
            return True

        current_shape_id, shape = shapes[0]
        invalid_positions = []
        for pos in random.sample(shape, len(shape)):
            place_res, info= can_place_star(star_positions, pos)
            if place_res:
                print(f"Placing a star in the {current_shape_id} region with the fewest legal positions.\n" + f"Attempting to place a star at position {(pos[0]+1,pos[1]+1)}\n")
                remaining_shapes = update_valid_positions(shapes[1:], star_positions)
                if len(remaining_shapes) > 0:
                    print(f"Position {(pos[0]+1,pos[1]+1)} has no other stars in its row, column, or diagonals, complying with the rules. Star positions have been updated. Recalculate the number of legal positions in remaining regions, continue to select the {remaining_shapes[0][0]} region with the fewest legal positions to place a star\n")
                else:
                    print(f"Position {(pos[0]+1,pos[1]+1)} has no other stars in its row, column, or diagonals, complying with the rules. There are no remaining regions, it seems we have found a star placement solution that satisfies all conditions.\n")
                star_positions.append(pos)
                print(f"Current star positions:\n{print_stars(star_positions,original_grid)}\n" + "Current grid:\n" + print_grid_in_kor(grid, star_positions))
                attempts += 1

                
                if backtrack(remaining_shapes, last_shape_id=current_shape_id):
                    return True

                print(f"Removing the star at position {(pos[0]+1,pos[1]+1)} because no suitable placement cell can be found in subsequent regions\n")
                star_positions.remove(pos)
                print(f"Current star positions:\n{print_stars(star_positions,original_grid)}\n" + "Current grid:\n" + print_grid_in_kor(grid, star_positions))
            else:
                if random.random() < 0.2 and info[0] == 'diagonally adjacent':
                    print(f"Placing a star in the {current_shape_id} region with the fewest legal positions.\n" + f"Attempting to place a star at position {(pos[0]+1,pos[1]+1)}\n")
                    print(f'{(pos[0]+1,pos[1]+1)} is invalid because it is {info[0]} with star {(info[1][0] + 1, info[1][1] + 1)}. Trying again to place a star in the {current_shape_id} region\n')
                    attempts += 1
                elif random.random() < 0.1 and not info[0] == 'diagonally adjacent':
                    print(f"Placing a star in the {current_shape_id} region with the fewest legal positions.\n" + f"Attempting to place a star at position {(pos[0]+1,pos[1]+1)}\n")
                    print(f'{(pos[0]+1,pos[1]+1)} is invalid because it is {info[0]} with star {(info[1][0] + 1, info[1][1] + 1)}. Trying again to place a star in the {current_shape_id} region\n')
                    attempts += 1
                    
                invalid_positions.append(pos)

        if invalid_positions:
            print(f"All positions in the {current_shape_id} region {','.join(f'({pos[0]+1}, {pos[1]+1})' for pos in invalid_positions)} conflict with existing star rules, no legal positions, cannot place a star\n")
        if last_shape_id:
            print(f"Currently, the {current_shape_id} region has no remaining legal star placement positions, backtracking to the {last_shape_id} region\n")
        else:
            # print_conclude("No solution found")
            pass
        return False

    if backtrack(shapes, None):
        return print_stars(star_positions,original_grid), attempts
    else:
        print("We have tried all possible solutions.No solution found.I'm afraid there is no solution for this puzzle.\n")
        return None, None


if __name__ == "__main__":
    # Example grid, numbers represent region IDs
    from grids import grids
    grid = grids[-2]
    star_positions, attempts = solve_star_battle(grid)
    print(f"Number of attempts: {attempts}")