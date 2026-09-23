def solve_campsite(grid, row_constraints, col_constraints):
    """
    Solve the 'Campsite' puzzle using backtracking.

    :param grid: A list of lists of characters ('T' or 'X') of size n×m.
    :param row_constraints: A list of length n with the required tent count per row.
    :param col_constraints: A list of length m with the required tent count per column.
    :return: A list of lists representing a solution (replacing 'X' with either 'C' or 'X'),
             or None if no solution exists.
    """

    n = len(grid)
    m = len(grid[0])

    # Copy the grid to store the solution without mutating the original
    solution = [row[:] for row in grid]

    # Track how many tents are currently placed in each row/column
    current_row_tents = [0] * n
    current_col_tents = [0] * m

    # Precompute which cells are adjacent to at least one tree
    # because a tent can only go orthogonally adjacent to a tree
    adjacent_to_tree = [[False]*m for _ in range(n)]
    directions_orth = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(n):
        for c in range(m):
            if grid[r][c] == 'T':
                for dr, dc in directions_orth:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < n and 0 <= cc < m and grid[rr][cc] == 'X':
                        adjacent_to_tree[rr][cc] = True

    # Helper function to check if placing a tent at (r, c) violates adjacency constraints
    def can_place_tent(r, c):
        # If the cell is not empty or not adjacent to a tree, cannot place a tent
        if solution[r][c] != 'X' or not adjacent_to_tree[r][c]:
            return False

        # Check row/col constraints if adding one more tent is still valid
        if current_row_tents[r] + 1 > row_constraints[r]:
            return False
        if current_col_tents[c] + 1 > col_constraints[c]:
            return False

        # Tents cannot be adjacent (orth or diag) to any existing tent
        # We'll check the 8 directions around (r, c)
        directions_all = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        for dr, dc in directions_all:
            rr, cc = r+dr, c+dc
            if 0 <= rr < n and 0 <= cc < m and solution[rr][cc] == 'C':
                return False

        return True

    # A list of all grid coordinates we might try to fill
    # Sorting them can sometimes help performance, but plain is fine, too.
    all_cells = [(r, c) for r in range(n) for c in range(m)]
    history = []
    
    # We'll do a simple backtracking approach that tries to place a tent or not
    def backtrack(idx=0):
        if idx == len(all_cells):
            history.append(f"All positions have been tried. Check if the current solution satisfies the row constraints and the col constraints.\n")
            is_satisfy = all(current_row_tents[r] == row_constraints[r] for r in range(n)) and  all(current_col_tents[c] == col_constraints[c] for c in range(m))
            if  is_satisfy:
                history.append("The current solution satisfy all the conditions. Find the solution!\n")
            else:
                history.append("The current solution does not satisfy all the conditions.\n")            
            return is_satisfy

        r, c = all_cells[idx]
        history.append(f"Select ({r},{c}), check if the tent can be placed in this location.\n")

        # -----------------------------------------------------------
        # OPTION 1: Place a tent in (r, c) if it's a valid move
        # -----------------------------------------------------------
        if can_place_tent(r, c):
            history.append(f"Tent can be placed at ({r},{c}). Place a tent at ({r},{c}).\n") 
            solution[r][c] = 'C'
            current_row_tents[r] += 1
            current_col_tents[c] += 1

            if backtrack(idx + 1):
                return True

            history.append(f"Since current solution fails. Change the current location ({r},{c}) from a tent(C) to an open space(X). Then retry.\n")
            # Backtrack (undo placement)
            solution[r][c] = 'X'
            current_row_tents[r] -= 1
            current_col_tents[c] -= 1
        else:
            history.append(f"No tents can be placed in ({r},{c}).\n")
        
        # -----------------------------------------------------------
        # OPTION 2: Skip placing a tent at (r, c)
        # -----------------------------------------------------------
        if backtrack(idx + 1):
            return True
        history.append(f"Since there is no tent placed at the current location ({r},{c}), we backtrack to the previous location.\n")

        # If neither placing nor skipping yields a solution, fail here
        return False

    # Kick off the backtracking
    if backtrack():
        history.append('\nThe final answer is: '+'{'+'\n'.join([' '.join(row) for row in solution]) + '}')
        steps = ''.join(history) 
        return solution, steps
    else:
        return None

# -----------------------------------------------------------------
# Example usage:
#
# Suppose you have the puzzle:
if __name__ == "__main__":
    grid = [
        ['X','X','X','X','X','T','X','X','T','X'],
        ['X','T','X','X','X','X','X','X','X','X'],
        ['X','X','X','T','T','X','X','X','X','X'],
        ['T','X','X','X','X','X','T','T','X','X'],
        ['X','T','X','X','T','X','X','X','X','X']
    ]
    row_constraints = [3, 1, 3, 0, 3]
    col_constraints = [2, 1, 1, 1, 1, 0, 2, 1, 0, 1]

    solved, steps = solve_campsite(grid, row_constraints, col_constraints)
    if solved:
        # Format the output as requested
        print(steps)
    else:
        print("No solution found (but puzzle is assumed to have a unique solution).")
    #
    # You can adapt this to parse your input format, then call `solve_campsite`
    # to get the solution, and finally output the result in the requested
    # "[[row1, row2, ...]]" string format.
